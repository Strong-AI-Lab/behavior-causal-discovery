
import argparse
import os
import re
import pandas as pd
import numpy as np

from src.data.dataset import SeriesDataset, DiscriminatorDataset
from src.data.format_data import PandasFormatterEnsemble
from src.data.constants import MASKED_VARIABLES
from src.model.model import TSLinearCausal, DISCRIMINATORS, MODELS
from src.evaluate.evaluation import direct_prediction_accuracy, generate_series
from src.evaluate.visualisation import generate_time_occurences, generate_sankey

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl



# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('save', type=str, help='Load the causal graph from a save folder.')
parser.add_argument('--discriminator_save', type=str, default=None, help='If provided, loads the discriminator from a save folder instead of running the algorithm again.')
parser.add_argument('--discriminator_type', type=str, default="lstm", help=f'Type of discriminator to use. Options: {",".join(DISCRIMINATORS.keys())}.')
parser.add_argument('--model_type', type=str, default="causal", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
args = parser.parse_args()

save = args.save
print(f"Arguments: save={save}")
if args.discriminator_save is not None:
    discriminator_save = args.discriminator_save
    print(f"Arguments: discriminator_save={discriminator_save}")

assert args.discriminator_type in DISCRIMINATORS.keys(), f"Discriminator type {args.discriminator_type} not supported. Options: {','.join(DISCRIMINATORS.keys())}."
assert args.model_type in MODELS.keys(), f"Model type {args.model_type} not supported. Options: {','.join(MODELS.keys())}."


# Read data
test_data_files = [name for name in os.listdir('data/test') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
test_data = [pd.read_csv(f'data/test/{name}') for name in test_data_files]

if args.discriminator_save is None:
    train_data_files = [name for name in os.listdir('data/train') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
    train_data = [pd.read_csv(f'data/train/{name}') for name in train_data_files]


# Set constants
TAU_MAX = 5


# Format data
test_formatter = PandasFormatterEnsemble(test_data)
test_sequences, *_ = test_formatter.format(event_driven=True)
test_sequences = {i: sequence for i, sequence in enumerate(test_sequences)}
variables = test_formatter.get_formatted_columns()

if args.discriminator_save is None:
    train_formatter = PandasFormatterEnsemble(train_data)
    train_sequences, *_ = train_formatter.format(event_driven=True)
    train_sequences = {i: sequence for i, sequence in enumerate(train_sequences)}

    assert variables == train_formatter.get_formatted_columns(), f"Test and train data have different variables: {variables} vs {train_formatter.get_formatted_columns()}"

num_var = len(variables)
print(f"Graph with {num_var} variables: {variables}.")


# Create dataset
test_dataset = SeriesDataset(test_sequences, lookback=TAU_MAX+1)
if args.discriminator_save is None:
    train_dataset = SeriesDataset(train_sequences, lookback=TAU_MAX+1)


# Get model
print(f"Save provided. Loading {args.model_type} model from {save}...")
if args.model_type == "causal":
    print("Causal model detected.")
    val_matrix = np.load(f'{save}/val_matrix.npy')
    val_matrix = torch.nan_to_num(torch.from_numpy(val_matrix).float())

    graph = np.load(f'{save}/graph.npy')
    graph[np.where(graph != "-->")] = "0"
    graph[np.where(graph == "-->")] = "1"
    graph = graph.astype(np.int64)
    graph = torch.from_numpy(graph).float()

    model = TSLinearCausal(num_var, TAU_MAX+1, weights=graph*val_matrix)
else:
    print("Parametric model detected.")
    model = MODELS[args.model_type].load_from_checkpoint(save)
    model.eval()


# Mask context variables for predition
masked_idxs = [variables.index(var) for var in MASKED_VARIABLES]
print(f"Masking {len(masked_idxs)} variables: {MASKED_VARIABLES}")


# Compute series
MIN_LENGTH = 30

test_series = generate_series(model, test_dataset, num_var, masked_idxs)
nb_test_series = len(test_series)
test_series = {k: v for k, v in test_series.items() if len(v) >= MIN_LENGTH}
print(f"Removed {nb_test_series - len(test_series)}/{nb_test_series} series with length < {MIN_LENGTH}.")

if args.discriminator_save is None:
    train_series = generate_series(model, train_dataset, num_var, masked_idxs)
    nb_train_series = len(train_series)
    train_series = {k: v for k, v in train_series.items() if len(v) >= MIN_LENGTH}
    print(f"Removed {nb_train_series - len(train_series)}/{nb_train_series} series with length < {MIN_LENGTH}.")



# Build discriminator
discriminator = DISCRIMINATORS[args.discriminator_type](num_var-len(MASKED_VARIABLES), TAU_MAX+1)
test_discr_dataset = DiscriminatorDataset(test_series)
test_loader = DataLoader(test_discr_dataset, batch_size=4, shuffle=True)
trainer = pl.Trainer(
        max_epochs=3,
        devices=[0], 
        accelerator="gpu")


if args.discriminator_save is not None:
    print(f"Discriminator save provided. Loading results from {discriminator_save}...")
    discriminator.load_state_dict(torch.load(f'{discriminator_save}/discriminator.pt'))
else:
    # Train discriminator
    print("Training discriminator...")
    train_discr_dataset = DiscriminatorDataset(train_series)
    train_loader = DataLoader(train_discr_dataset, batch_size=4, shuffle=True)
    trainer.fit(discriminator, train_loader)

# Test model against discriminator
predictions = trainer.predict(discriminator, test_loader)
accuracy = []
for y_pred, (x, y) in zip(predictions, test_loader):
    accuracy.append((y_pred.argmax(dim=-1) == y).float().mean())

print(f"Discriminator accuracy: {torch.tensor(accuracy).mean()}")
