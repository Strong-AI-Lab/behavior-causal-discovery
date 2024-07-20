
import argparse
import os
import re

from data.dataset import SeriesDataset
from data.structure.chronology import Chronology
from data.structure.loaders import DynamicSeriesLoader, DynamicGraphSeriesLoader
from model.dynamics_model import DYNAMIC_MODELS
from model.graph_dynamics_model import GRAPH_DYNAMIC_MODELS

import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
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
parser.add_argument('--save', type=str, default=None, help='If provided, loads the model from a save. The save can be a `model.ckpt` file. If the model_type if `causal_*`, a save folder from a causal_discovery run can also be used.')
parser.add_argument('--wandb_project', type=str, default=None, help='If specified, logs the run to wandb under the specified project.')
parser.add_argument('--force_data_computation', action="store_true", help='If specified, forces the computation of the force data from the raw data.')
args = parser.parse_args()

assert args.model_type in MODELS.keys(), f"Model type {args.model_type} not supported. Options: {','.join(MODELS.keys())}."

is_graph_model = False
structure_savefile = "train_chronology_dynamics.json"
dataset_savefile = "train_data_dynamics.pt"
if args.model_type in GRAPH_DYNAMIC_MODELS.keys():
        is_graph_model = True
        dataset_savefile = "graph_" + dataset_savefile


# Set constants
TAU_MAX = 5


if not args.force_data_computation and os.path.exists(f"data/gen/{dataset_savefile}"):
        print(f"Loading dataset from data/gen/{dataset_savefile}...")
        train_dataset = SeriesDataset.load(f"data/gen/{dataset_savefile}")

elif not args.force_data_computation and os.path.exists(f"data/gen/{structure_savefile}"):
        print(f"No dataset found in data/gen. Data structure found and loaded from data/gen/{structure_savefile}. Re-computing the dataset from structure...")

        # Create dataset
        chronology = Chronology.deserialize(f"data/gen/{structure_savefile}")
        structure_loader = DynamicGraphSeriesLoader(lookback=TAU_MAX+1, skip_stationary=True) if is_graph_model else DynamicSeriesLoader(lookback=TAU_MAX+1, skip_stationary=True)
        train_dataset = SeriesDataset(chronology=chronology, struct_loader=structure_loader)

        #Save dataset
        train_dataset.save(f"data/gen/{dataset_savefile}")

else:
        if not os.path.exists(f"data/gen/{dataset_savefile}"):
                print("No dataset or data structure found in data/gen. Generating dataset...")
        if args.force_data_computation:
                print("Forced data computation. (Re)computing dataset...")

        # Create structure
        chronology = Chronology.create([f'data/train/{name}' for name in os.listdir('data/train') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)])
        
        # Save structure
        os.makedirs("data/gen", exist_ok=True)
        chronology.serialize(f"data/gen/{structure_savefile}")

        # Create dataset
        structure_loader = DynamicGraphSeriesLoader(lookback=TAU_MAX+1, skip_stationary=True) if is_graph_model else DynamicSeriesLoader(lookback=TAU_MAX+1, skip_stationary=True)
        train_dataset = SeriesDataset(chronology=chronology, struct_loader=structure_loader)

        # Save dataset
        train_dataset.save(f"data/gen/{dataset_savefile}")


# Build data loader
dataset = torch.utils.data.random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset))], generator=torch.Generator().manual_seed(1))
if is_graph_model:
        train_loader = GraphDataLoader(dataset[0], batch_size=64, shuffle=True)
        val_loader = GraphDataLoader(dataset[1], batch_size=64, shuffle=False)
else:
        train_loader = DataLoader(dataset[0], batch_size=64, shuffle=True)
        val_loader = DataLoader(dataset[1], batch_size=64, shuffle=False)



# Build model
if args.save is None:
        print(f"No save provided. Building {args.model_type} model...")
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