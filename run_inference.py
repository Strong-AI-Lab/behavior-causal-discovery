
import argparse
import os
import re
import pandas as pd
import numpy as np

from src.data.dataset import SeriesDataset
from src.data.format_data import PandasFormatterEnsemble
from src.data.constants import MASKED_VARIABLES
from src.model.model import TSLinearCausal
from src.evaluate.evaluation import direct_prediction_accuracy, generate_series
from src.evaluate.visualisation import generate_time_occurences, generate_sankey

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
data_files = [name for name in os.listdir('data/test') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
data = [pd.read_csv(f'data/test/{name}') for name in data_files]
print(data)


# Set constants
TAU_MAX = 5


# Format data
formatter = PandasFormatterEnsemble(data)
sequences = formatter.format(event_driven=True)
sequences = {i: sequence for i, sequence in enumerate(sequences)}
variables = formatter.get_formatted_columns()
num_var = len(variables)
print(f"Graph with {num_var} variables: {variables}.")


# Create dataset
dataset = SeriesDataset(sequences, tau_max=TAU_MAX+1)


# Get model
print(f"Save provided. Loading results from {save}...")
val_matrix = np.load(f'{save}/val_matrix.npy')
val_matrix = torch.nan_to_num(torch.from_numpy(val_matrix).float())

graph = np.load(f'{save}/graph.npy')
graph[np.where(graph != "-->")] = "0"
graph[np.where(graph == "-->")] = "1"
graph = graph.astype(np.int64)
graph = torch.from_numpy(graph).float()

model = TSLinearCausal(num_var, TAU_MAX+1, weights=graph*val_matrix)
random_loader = DataLoader(dataset, batch_size=4, shuffle=True)


# Mask context variables for predition
masked_idxs = [variables.index(var) for var in MASKED_VARIABLES]
print(f"Masking {len(masked_idxs)} variables: {MASKED_VARIABLES}")



# Evaluate model

# Compute direct prediction accuracy
acc = direct_prediction_accuracy(model, random_loader, num_var, masked_idxs)
print(f"Direct Prediction Accuracy: {acc}")


# Compute series prediction metrics
series = generate_series(model, dataset, num_var, masked_idxs)
nb_series = len(series)
print(f"Generated {nb_series} series.")

MIN_LENGTH = 30
predicted_variable_names = [re.sub("_", " ", re.sub(r"\(.*\)", "", v)) for i, v in enumerate(variables) if i not in masked_idxs]
nb_variables = len(predicted_variable_names)
series = {k: v for k, v in series.items() if len(v) >= MIN_LENGTH}
print(f"Removed {nb_series - len(series)}/{nb_series} series with length < {MIN_LENGTH}.")


# Visualise time occurences
generate_time_occurences(series, predicted_variable_names, save, nb_variables, MIN_LENGTH)

# Visualise Sankey flows
generate_sankey(series, predicted_variable_names, save, nb_variables, MIN_LENGTH)

print(f"Figures saved in results/{save.split('/')[-1]}.")


# ARIMA test (e.g. https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/, https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)

# Granger test (e.g. https://www.rdocumentation.org/packages/lmtest/versions/0.9-35/topics/grangertest)

# Kolmogorov-Smirnov test (e.g. https://towardsdatascience.com/how-to-compare-two-distributions-in-practice-8c676904a285)