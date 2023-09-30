
import argparse
import os
import re
import pandas as pd
import numpy as np

from src.data.format_data import PandasFormatterEnsemble
from src.data.dataset import SeriesDataset
from src.model.model import MODELS

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl



# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('--mode_type',ype=str, default="lstm", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
args = parser.parse_args()

assert args.model_type in MODELS.keys(), f"Model type {args.model_type} not supported. Options: {','.join(MODELS.keys())}."
assert args.model_type != "causal", f"Model type {args.model_type} does not support training. Use `run_discovery.py` instead."

# Read data
test_data_files = [name for name in os.listdir('data/test') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
test_data = [pd.read_csv(f'data/test/{name}') for name in test_data_files]

train_data_files = [name for name in os.listdir('data/train') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
train_data = [pd.read_csv(f'data/train/{name}') for name in train_data_files]


# Set constants
TAU_MAX = 5


# Format data
test_formatter = PandasFormatterEnsemble(test_data)
test_sequences = test_formatter.format(event_driven=True)
test_sequences = {i: sequence for i, sequence in enumerate(test_sequences)}
variables = test_formatter.get_formatted_columns()

train_formatter = PandasFormatterEnsemble(train_data)
train_sequences = train_formatter.format(event_driven=True)
train_sequences = {i: sequence for i, sequence in enumerate(train_sequences)}

assert variables == train_formatter.get_formatted_columns(), f"Test and train data have different variables: {variables} vs {train_formatter.get_formatted_columns()}"

num_var = len(variables)
print(f"Graph with {num_var} variables: {variables}.")


# Create dataset
test_dataset = SeriesDataset(test_sequences, tau_max=TAU_MAX+1)
train_dataset = SeriesDataset(train_sequences, tau_max=TAU_MAX+1)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


# Train model
model = MODELS[args.model_type](num_var, tau_max=TAU_MAX+1)

trainer = pl.Trainer(
        max_epochs=3,
        devices=[0], 
        accelerator="gpu")

trainer.fit(model, train_loader)

# Test model
predictions = trainer.predict(model, test_loader)
accuracy = []
for y_pred, (x, y, i) in zip(predictions, test_loader):
    accuracy.append((y_pred.argmax(dim=-1) == y).float().mean())

print(f"Model accuracy: {torch.tensor(accuracy).mean()}")