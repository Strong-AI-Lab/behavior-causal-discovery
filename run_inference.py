
import argparse
import pandas as pd
import numpy as np
from format_data import PandasFormatter
import tqdm

from dataset import SeriesDataset
from model import TSLinearCausal

import torch
from torch.utils.data import DataLoader


# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('save', type=str, help='Load the causal graph from a save folder.')
args = parser.parse_args()

save = args.save
print(f"Arguments: save={save}")


# Read data
data = pd.read_csv('data/cam2_2022_10_20_GH200023.csv')
print(data)


# Set constants
tau_max = 5


# Format data
formatter = PandasFormatter(data)
sequences = formatter.format(event_driven=True)
sequences = {i: sequence for i, sequence in enumerate(sequences)}
variables = formatter.get_formatted_columns()
num_var = len(variables)
print(f"Graph with {num_var} variables: {variables}.")


# Create dataset
dataset = SeriesDataset(sequences, tau_max=tau_max+1)


# Get model
print(f"Save provided. Loading results from {save}...")
val_matrix = np.load(f'{save}/val_matrix.npy')
val_matrix = torch.nan_to_num(torch.from_numpy(val_matrix).float())

graph = np.load(f'{save}/graph.npy')
graph[np.where(graph != "-->")] = "0"
graph[np.where(graph == "-->")] = "1"
graph = graph.astype(np.int64)
graph = torch.from_numpy(graph).float()

model = TSLinearCausal(num_var, tau_max+1, weights=graph*val_matrix)
loader = DataLoader(dataset, batch_size=4, shuffle=True)


# Create mask
masked_variables = ['background', 'waiting_area', 'door', 'close_neighbour_foraging', 'close_neighbour_background', 'close_neighbour_waiting_area', 'close_neighbour_door', 'distant_neighbour_foraging', 'distant_neighbour_background', 'distant_neighbour_waiting_area', 'distant_neighbour_door', 'close_neighbour_foraging', 'close_neighbour_raised_guarding_(vigilant)', 'close_neighbour_huddle', 'close_neighbour_high_sitting/standing_(vigilant)', 'close_neighbour_moving', 'close_neighbour_playfight', 'close_neighbour_low_sitting/standing_(stationary)', 'close_neighbour_groom', 'close_neighbour_human_interaction', 'close_neighbour_interacting_with_foreign_object', 'close_neighbour_dig_burrow', 'distant_neighbour_foraging', 'distant_neighbour_raised_guarding_(vigilant)', 'distant_neighbour_huddle', 'distant_neighbour_high_sitting/standing_(vigilant)', 'distant_neighbour_moving', 'distant_neighbour_playfight', 'distant_neighbour_low_sitting/standing_(stationary)', 'distant_neighbour_groom', 'distant_neighbour_human_interaction', 'distant_neighbour_interacting_with_foreign_object', 'distant_neighbour_dig_burrow']
masked_idxs = [variables.index(var) for var in masked_variables]
print(f"Masking {len(masked_idxs)} variables: {masked_variables}")


# Evaluate model
acc = torch.tensor(0.0)
for i, (x, y) in enumerate(tqdm.tqdm(loader)):
    # Make prediction
    y_pred = model(x)

    # Remove masked variables
    y_pred = y_pred[:,:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
    y = y[:,:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]

    # Calculate accuracy
    acc = acc * i / (i+1) + (y_pred.argmax(dim=-1) == y.argmax(dim=-1)).float().mean() / (i+1)

print(f"Accuracy: {acc}")