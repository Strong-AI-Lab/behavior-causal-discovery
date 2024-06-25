
import argparse
import os
import re
import pandas as pd
import numpy as np

from src.data.format_data import PandasFormatterEnsemble
from src.data.dataset import DynamicSeriesDataset, DynamicGraphSeriesDataset
from src.model.dynamics_model import DYNAMIC_MODELS
from src.model.graph_dynamics_model import GRAPH_DYNAMIC_MODELS
from src.dynamics.solver import DynamicsSolver

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


MODELS = {
    **DYNAMIC_MODELS,
    **GRAPH_DYNAMIC_MODELS
}


# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('--model_type',type=str, default="dynamical_lstm", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser.add_argument('--save', type=str, default=None, help='If provided, loads the model from a save. The save can be a `model.ckpt` file. If the model_type if `causal_*`, a save folder from a causal_discovery run can als be used.')
parser.add_argument('--wandb_project', type=str, default=None, help='If specified, logs the run to wandb under the specified project.')
parser.add_argument('--force_data_computation', action="store_true", help='If specified, forces the computation of the force data from the raw data.')
args = parser.parse_args()

assert args.model_type in MODELS.keys(), f"Model type {args.model_type} not supported. Options: {','.join(MODELS.keys())}."

is_graph_model = False
savefile = "data_dynamics.pt"
if args.model_type in GRAPH_DYNAMIC_MODELS.keys():
        is_graph_model = True
        savefile = "graph_" + savefile


# Set constants
TAU_MAX = 5
LOW_FILTER = 0.075


if not args.force_data_computation and os.path.exists(f"data/gen/{savefile}"):
        print(f"Loading dataset from data/gen/{savefile}...")
        train_dataset = torch.load(f"data/gen/{savefile}")

else:
        if not os.path.exists(f"data/gen/{savefile}"):
                print("No dataset found in data/gen. Generating dataset...")
        if args.force_data_computation:
                print("Forced data computation. (Re)computing dataset...")


        # Read data
        train_data_files = [name for name in os.listdir('data/train') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
        train_data = [pd.read_csv(f'data/train/{name}') for name in train_data_files]

        # Format data
        train_formatter = PandasFormatterEnsemble(train_data)
        res = train_formatter.format(output_format="dataclass")
        train_sequences = res.movements
        train_graph = res.neighbor_graphs
        train_sequences = {ind : coords.to_numpy(dtype=np.float64).tolist() for ind, coords in train_sequences.items()}

        # Create dataset
        if is_graph_model:
                train_dataset = DynamicGraphSeriesDataset(train_sequences, adjacency_list=train_graph, lookback=TAU_MAX+1, target_offset_start=0, target_offset_end=0)
        else:
                train_dataset = DynamicSeriesDataset(train_sequences, lookback=TAU_MAX+1, target_offset_start=0, target_offset_end=0) # no offset as we want to predict the force applied on the current step

        # Save dataset
        os.makedirs("data/gen", exist_ok=True)
        torch.save(train_dataset,f"data/gen/{savefile}")


# Build data loader
dataset = torch.utils.data.random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset))], generator=torch.Generator().manual_seed(1))
train_loader = DataLoader(dataset[0], batch_size=64, shuffle=True)
val_loader = DataLoader(dataset[1], batch_size=64, shuffle=False)



# Build model
if args.save is None:
        model = MODELS[args.model_type](lookback=TAU_MAX+1)
else:
        print(f"Save provided. Loading {args.model_type} model from {args.save}...")
        model = MODELS[args.model_type].load_from_checkpoint(args.save, lookback=TAU_MAX+1)


# Train model
trainer = pl.Trainer(
        max_epochs=20,
        devices=[0],
        accelerator="gpu",
        logger=WandbLogger(name=f"{args.model_type}_train", project=args.wandb_project) if args.wandb_project else None,
)

trainer.fit(model, train_loader, val_loader)