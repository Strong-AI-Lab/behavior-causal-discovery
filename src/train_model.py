
import argparse
import os
import re
import numpy as np

from data.dataset import SeriesDataset
from data.structure.chronology import Chronology
from data.structure.loaders import BehaviourSeriesLoader
from data.constants import MASKED_VARIABLES, VECTOR_COLUMNS
from model.behaviour_model import BEHAVIOUR_MODELS
from model.causal_graph_formatter import CausalGraphFormatter

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
variables = VECTOR_COLUMNS
num_variables = len(variables)

structure_savefile = "train_chronology_behaviours.json"
dataset_savefile = "train_data_behaviour.pt"

if not args.force_data_computation and os.path.exists(f"data/gen/{dataset_savefile}"):
        print(f"Loading dataset from data/gen/{dataset_savefile}...")
        train_dataset = SeriesDataset.load(f"data/gen/{dataset_savefile}")

elif not args.force_data_computation and os.path.exists(f"data/gen/{structure_savefile}"):
        print(f"No dataset found in data/gen. Data structure found and loaded from data/gen/{structure_savefile}. Re-computing the dataset from structure...")

        # Create dataset
        chronology = Chronology.deserialize(f"data/gen/{structure_savefile}")
        structure_loader = BehaviourSeriesLoader(lookback=TAU_MAX+1, skip_stationary=True)
        train_dataset = SeriesDataset(chronology=chronology, struct_loader=structure_loader)

        #Save dataset
        train_dataset.save(f"data/gen/{dataset_savefile}")

else:
        if not os.path.exists(f"data/gen/{dataset_savefile}"):
                print("No dataset or data structure found in data/gen. Generating dataset...")
        if args.force_data_computation:
                print("Forced data computation. (Re)computing dataset...")

        # Create structure
        chronology = Chronology.create([f'data/train/{name}' for name in os.listdir('data/train') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)], fix_errors=True, filter_null_state_trajectories=True)

        # Save structure
        os.makedirs("data/gen", exist_ok=True)
        chronology.serialize(f"data/gen/{structure_savefile}")

        # Create dataset
        structure_loader = BehaviourSeriesLoader(lookback=TAU_MAX+1, skip_stationary=True)
        train_dataset = SeriesDataset(chronology=chronology, struct_loader=structure_loader)

        # Save dataset
        train_dataset.save(f"data/gen/{dataset_savefile}")



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
                                        filtered_values = CausalGraphFormatter(graph, val_matrix).low_filter(LOW_FILTER)
                                        val_matrix = filtered_values.get_val_matrix()
                                        graph = filtered_values.get_graph()
                                elif f == "neighbor_effect":
                                        filtered_values = CausalGraphFormatter(graph, val_matrix).var_filter([], [variables.index(v) for v in variables if v.startswith('close_neighbour_') or v.startswith('distant_neighbour_')])
                                        val_matrix = filtered_values.get_val_matrix()
                                        graph = filtered_values.get_graph()
                                elif f == "corr":
                                        filtered_values = CausalGraphFormatter(graph, val_matrix).corr_filter()
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