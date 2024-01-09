
import argparse
import os
import re
import pandas as pd
import numpy as np

from src.data.format_data import PandasFormatterEnsemble
from src.data.dataset import SeriesDataset
from src.model.dynamics_model import DYNAMIC_MODELS
from src.dynamics.solver import DynamicsSolver

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


MODELS = {
    **DYNAMIC_MODELS,
}


# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('--model_type',type=str, default="dynamical_lstm", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser.add_argument('--save', type=str, default=None, help='If provided, loads the model from a save. The save can be a `model.ckpt` file. If the model_type if `causal_*`, a save folder from a causal_discovery run can als be used.')
args = parser.parse_args()

# assert args.model_type in MODELS.keys(), f"Model type {args.model_type} not supported. Options: {','.join(MODELS.keys())}."

# Read data
train_data_files = [name for name in os.listdir('data/train') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
train_data = [pd.read_csv(f'data/train/{name}') for name in train_data_files]


# Set constants
TAU_MAX = 5
LOW_FILTER = 0.075


# Format data
train_formatter = PandasFormatterEnsemble(train_data)
train_sequences = train_formatter.format(output_format="dataclass").movements
train_sequences = {ind : coords.to_numpy(dtype=np.float64).tolist() for ind, coords in train_sequences.items()}


# Create dataset
solver = DynamicsSolver(mass=1, dimensions=3)
def transform(sample):
        x, y, ind = sample
        y = solver.compute_acceleration(y.unsqueeze(0)) # target data is force applied on target step (t+1), corresponds to acceleration when setting mass=1
        y = y.squeeze(0)
        return x, y, ind

train_dataset = SeriesDataset(train_sequences, lookback=TAU_MAX+1, target_offset_start=1, target_offset_end=3, transform=transform) # add 2 to offset to compute acceleration of target step (t+1)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# # Build model
if args.save is None:
        model = MODELS[args.model_type](lookback=TAU_MAX+1)
else:
        print(f"Save provided. Loading {args.model_type} model from {args.save}...")
        model = MODELS[args.model_type].load_from_checkpoint(args.save, lookback=TAU_MAX+1)


# Train model
trainer = pl.Trainer(
        max_epochs=10,
        devices=[0],
        accelerator="gpu")

trainer.fit(model, train_loader)