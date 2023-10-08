
import argparse
import os
import re
import pandas as pd
import numpy as np
import tqdm

from src.data.dataset import SeriesDataset, DiscriminatorDataset
from src.data.format_data import PandasFormatterEnsemble, ResultsFormatter
from src.data.constants import MASKED_VARIABLES
from src.model.model import TSLinearCausal, DISCRIMINATORS, MODELS
from src.evaluate.evaluation import generate_series, generate_series_community

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
parser.add_argument('--community', action='store_true', help='If provided, the discriminator is trained on community-generated data.')
parser.add_argument('--filter', type=str, default=None, help='If provided, filters the causal graph to only include the most significant links. Options: ' + 
                                                                '"low"  : remove links with low values; ' +
                                                                '"neighbor_effect" : remove links to neighbors, ' + 
                                                                '"corr" : remove correlations without causation. ' +
                                                                'Multiple filters can be applied by separating them with a comma.')
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
LOW_FILTER = 0.075


# Format data
test_formatter = PandasFormatterEnsemble(test_data)
test_sequences, test_true_ind_sequences, test_neighbor_graphs = test_formatter.format(event_driven=True)
test_sequences = {i: sequence for i, sequence in enumerate(test_sequences)}
variables = test_formatter.get_formatted_columns()

if args.discriminator_save is None:
    train_formatter = PandasFormatterEnsemble(train_data)
    train_sequences, train_true_ind_sequences, train_neighbor_graphs = train_formatter.format(event_driven=True)
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
    graph = np.load(f'{save}/graph.npy')

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

    model = TSLinearCausal(num_var, TAU_MAX+1, weights=graph*val_matrix)
else:
    print("Parametric model detected.")
    if torch.cuda.is_available():
        map_location=torch.device('cuda')
    else:
        map_location=torch.device('cpu')
    model = MODELS[args.model_type].load_from_checkpoint(save, num_var=num_var, lookback=TAU_MAX+1, map_location=map_location)


# Mask context variables for predition
masked_idxs = [variables.index(var) for var in MASKED_VARIABLES]
close_neighbor_idxs = [variables.index(var) for var in variables if var.startswith('close_neighbour_') and not var.endswith('_zone')]
distant_neighbor_idxs = [variables.index(var) for var in variables if var.startswith('distant_neighbour_') and not var.endswith('_zone')]
print(f"Masking {len(masked_idxs)} variables: {MASKED_VARIABLES}")


# Compute series
MIN_LENGTH = 30

with torch.no_grad():
    if args.community:
        test_community_dataset = SeriesDataset({ind: seq.to_numpy(dtype=np.float64) for ind, seq in test_true_ind_sequences.items()}, lookback=TAU_MAX+1)
        test_series = generate_series_community(model, test_community_dataset, test_neighbor_graphs, num_var, masked_idxs, close_neighbor_idxs, distant_neighbor_idxs)
    else:
        test_series = generate_series(model, test_dataset, num_var, masked_idxs)

    nb_test_series = len(test_series)
    test_series = {k: v for k, v in test_series.items() if len(v) >= MIN_LENGTH}
    print(f"Removed {nb_test_series - len(test_series)}/{nb_test_series} series with length < {MIN_LENGTH}.")

    if args.discriminator_save is None:
        if args.community:
            train_community_dataset = SeriesDataset({ind: seq.to_numpy(dtype=np.float64) for ind, seq in train_true_ind_sequences.items()}, lookback=TAU_MAX+1)
            train_series = generate_series_community(model, train_community_dataset, train_neighbor_graphs, num_var, masked_idxs, close_neighbor_idxs, distant_neighbor_idxs)
        else:
            train_series = generate_series(model, train_dataset, num_var, masked_idxs)
            nb_train_series = len(train_series)
            train_series = {k: v for k, v in train_series.items() if len(v) >= MIN_LENGTH}
            print(f"Removed {nb_train_series - len(train_series)}/{nb_train_series} series with length < {MIN_LENGTH}.")


# Build discriminator
test_discr_dataset = DiscriminatorDataset(test_series)
test_loader = DataLoader(test_discr_dataset, batch_size=64, shuffle=True)

if args.discriminator_save is not None:
    print(f"Discriminator save provided. Loading results from {discriminator_save}...")
    discriminator = DISCRIMINATORS[args.discriminator_type].load_from_checkpoint(save, num_var=num_var-len(MASKED_VARIABLES), lookback=TAU_MAX+1)
else:
    # Train discriminator
    print("Discriminator save not provided. Building a new discriminator.")
    discriminator = DISCRIMINATORS[args.discriminator_type](num_var-len(MASKED_VARIABLES), TAU_MAX+1)
    discriminator.train()


trainer = pl.Trainer(
        max_epochs=10,
        devices=[0], 
        accelerator="gpu")

if args.discriminator_save is None:
    print("Training discriminator...")
    train_discr_dataset = DiscriminatorDataset(train_series)
    train_loader = DataLoader(train_discr_dataset, batch_size=64, shuffle=True)
    trainer.fit(discriminator, train_loader)


# Test model against discriminator
accuracy = []
with torch.no_grad():
    for x, y in tqdm.tqdm(test_loader):
            x = x.to(discriminator.device)
            y = y.to(discriminator.device).unsqueeze(-1)
            
            # Make prediction
            y_pred = discriminator(x)
            y_pred = (y_pred > 0.5).int()

            # Calculate accuracy
            accuracy.extend(((y_pred) == y).float().tolist())

print(f"Discriminator accuracy: {torch.tensor(accuracy).mean()}")
