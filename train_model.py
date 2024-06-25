
import argparse
import os
import re
import pandas as pd
import numpy as np
import json

from src.data.format_data import PandasFormatterEnsemble, ResultsFormatter
from src.data.dataset import SeriesDataset
from src.model.behaviour_model import BEHAVIOUR_MODELS
from src.data.constants import MASKED_VARIABLES

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


MODELS = {
    **BEHAVIOUR_MODELS,
}


# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('--model_type',type=str, default="lstm", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser.add_argument('--save', type=str, default=None, help='If provided, loads the model from a save. The save can be a `model.ckpt` file. If the model_type if `causal_*`, a save folder from a causal_discovery run can also be used.')
parser.add_argument('--wandb_project', type=str, default=None, help='If specified, logs the run to wandb under the specified project.')
parser.add_argument('--force_data_computation', action="store_true", help='If specified, forces the computation of the force data from the raw data.')
parser.add_argument('--causal_graph', type=str, default="all", help='Only used when a save folder from a causal discovery run is loaded. Controls if the graph contains the edges the coefficients. Options: "all", "coefficients", "edges".')
parser.add_argument('--filter', type=str, default=None, help='If provided, filters the causal graph to only include the most significant links. Options: ' + 
                                                                '"low" : remove links with low values; ' +
                                                                '"neighbor_effect" : remove links to neighbors, ' + 
                                                                '"corr" : remove correlations without causation. ' +
                                                                'Multiple filters can be applied by separating them with a comma.')
args = parser.parse_args()

assert args.model_type in MODELS.keys(), f"Model type {args.model_type} not supported. Options: {','.join(MODELS.keys())}."
assert args.model_type != "causal", f"Model type {args.model_type} does not support training. Use `run_discovery.py` instead."

# Set constants
TAU_MAX = 5
LOW_FILTER = 0.075


if not args.force_data_computation and os.path.exists(f"data/gen/data_behaviour.pt") and os.path.exists(f"data/gen/variables_behaviour.json"):
        print("Loading dataset from data/gen/data_behaviour.pt...")
        train_dataset = torch.load("data/gen/data_behaviour.pt")
        variables = json.load(open("data/gen/variables_behaviour.json", "r"))
        num_variables = len(variables)

else:
        if not os.path.exists("data/gen/data_behaviour.pt"):
                print("No dataset found in data/gen. Generating dataset...")
        if not os.path.exists("data/gen/variables_behaviour.json"):
                print("No variables found in data/gen. Generating dataset...")
        if args.force_data_computation:
                print("Forced data computation. (Re)computing dataset...")

        # Read data
        test_data_files = [name for name in os.listdir('data/test') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
        test_data = [pd.read_csv(f'data/test/{name}') for name in test_data_files]

        train_data_files = [name for name in os.listdir('data/train') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
        train_data = [pd.read_csv(f'data/train/{name}') for name in train_data_files]

        # Format data
        test_formatter = PandasFormatterEnsemble(test_data)
        variables = test_formatter.get_formatted_columns()

        train_formatter = PandasFormatterEnsemble(train_data)
        train_sequences, *_ = train_formatter.format(event_driven=True)
        train_sequences = {i: sequence for i, sequence in enumerate(train_sequences)}

        assert variables == train_formatter.get_formatted_columns(), f"Test and train data have different variables: {variables} vs {train_formatter.get_formatted_columns()}"

        num_variables = len(variables)
        print(f"Graph with {num_variables} variables: {variables}.")

        # Create dataset
        train_dataset = SeriesDataset(train_sequences, lookback=TAU_MAX+1)

        # Save dataset
        os.makedirs("data/gen", exist_ok=True)
        torch.save(train_dataset,"data/gen/data_behaviour.pt")
        json.dump(variables, open("data/gen/variables_behaviour.json", "w"))



# Build data loader
dataset = torch.utils.data.random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset))], generator=torch.Generator().manual_seed(1))
train_loader = DataLoader(dataset[0], batch_size=64, shuffle=True)
val_loader = DataLoader(dataset[1], batch_size=64, shuffle=False)

# Mask context variables for predition
masked_idxs = [variables.index(var) for var in MASKED_VARIABLES]
print(f"Masking {len(masked_idxs)} variables: {MASKED_VARIABLES}")


# Build model
if args.save is None:
        model = MODELS[args.model_type](num_variables=num_variables, lookback=TAU_MAX+1, masked_idxs_for_training=masked_idxs)
else:
        print(f"Save provided. Loading {args.model_type} model from {args.save}...")
        if args.model_type.startswith("causal_"):
                print("Causal model detected.")
                val_matrix = np.load(f'{args.save}/val_matrix.npy')
                graph = np.load(f'{args.save}/graph.npy')

                if args.filter is not None:
                        for f in args.filter.split(","):
                                print(f"Filtering results using {f}...")
                                if f == 'low':
                                        filtered_values = ResultsFormatter(graph, val_matrix).low_filter(LOW_FILTER)
                                        val_matrix = filtered_values.get_val_matrix()
                                        graph = filtered_values.get_graph()
                                elif f == "neighbor_effect":
                                        filtered_values = ResultsFormatter(graph, val_matrix).var_filter([], [variables.index(v) for v in variables if v.startswith('close_neighbour_') or v.startswith('distant_neighbour_')])
                                        val_matrix = filtered_values.get_val_matrix()
                                        graph = filtered_values.get_graph()
                                elif f == "corr":
                                        filtered_values = ResultsFormatter(graph, val_matrix).corr_filter()
                                        val_matrix = filtered_values.get_val_matrix()
                                        graph = filtered_values.get_graph()
                                else:
                                        print(f"Filter {f} not recognised. Skipping filter...")

                val_matrix = torch.nan_to_num(torch.from_numpy(val_matrix).float())
                graph[np.where(graph != "-->")] = "0"
                graph[np.where(graph == "-->")] = "1"
                graph = graph.astype(np.int64)
                graph = torch.from_numpy(graph).float()

                if args.causal_graph == "all":
                        graph_weights=graph*val_matrix
                elif args.causal_graph == "coefficients":
                        graph_weights=val_matrix
                elif args.causal_graph == "edges":
                        graph_weights=graph
                else:
                        raise ValueError(f"causal_graph must be one of 'all', 'coefficients', 'edges'. Got {args.causal_graph}.")

                model = MODELS[args.model_type](num_variables=num_variables, lookback=TAU_MAX+1, graph_weights=graph_weights, masked_idxs_for_training=masked_idxs)
        else:
                print("Parametric model detected.")
                model = MODELS[args.model_type].load_from_checkpoint(args.save, num_variables=num_variables, lookback=TAU_MAX+1, masked_idxs_for_training=masked_idxs)


# Train model
trainer = pl.Trainer(
        max_epochs=10,
        devices=[0], 
        accelerator="gpu",
        
        logger=WandbLogger(name=f"{args.model_type}_train", project=args.wandb_project) if args.wandb_project else None,
        )

trainer.fit(model, train_loader, val_loader)