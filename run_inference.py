
import argparse
import os
import re
import pandas as pd
import numpy as np

from src.data.dataset import SeriesDataset
from src.data.format_data import PandasFormatterEnsemble
from src.data.constants import MASKED_VARIABLES
from src.model.model import TSLinearCausal, MODELS
from src.evaluate.evaluation import direct_prediction_accuracy, mutual_information, generate_series
from src.evaluate.visualisation import generate_time_occurences, generate_sankey, generate_clusters

import torch
from torch.utils.data import DataLoader


# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('save', type=str, help='Load the causal graph from a save folder.')
parser.add_argument('--model_type', type=str, default="causal", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
args = parser.parse_args()

save = args.save
print(f"Arguments: save={save}")
if save.endswith('/'):
    save = save[:-1]


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
random_loader = DataLoader(dataset, batch_size=4, shuffle=True)


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
    if torch.cuda.is_available():
        map_location=torch.device('cuda')
    else:
        map_location=torch.device('cpu')
    model = MODELS[args.model_type].load_from_checkpoint(save, num_var=num_var, tau_max=TAU_MAX+1, map_location=map_location)

    save_split = save.split('/')
    save = "/".join(save_split[:-3] + [save_split[-3] + "_" + save_split[-1][:-5]]) # Remove .ckpt from save path and concact with version to build results directory



# Mask context variables for predition
masked_idxs = [variables.index(var) for var in MASKED_VARIABLES]
print(f"Masking {len(masked_idxs)} variables: {MASKED_VARIABLES}")



# Evaluate model


model.eval()
with torch.no_grad():
    # Compute direct prediction accuracy
    acc = direct_prediction_accuracy(model, random_loader, num_var, masked_idxs)
    print(f"Direct Prediction Accuracy: {acc}")

    # Compute conditional mutual information
    cmi = mutual_information(model, random_loader, num_var, masked_idxs)
    print(f"Mutual Information: {cmi}")


    # Compute series prediction metrics
    series = generate_series(model, dataset, num_var, masked_idxs)
    nb_series = len(series)
    print(f"Generated {nb_series} series.")

    MIN_LENGTH = 30
    series = {k: v for k, v in series.items() if len(v) >= MIN_LENGTH}
    print(f"Removed {nb_series - len(series)}/{nb_series} series with length < {MIN_LENGTH}.")


    # Visualise time occurences
    predicted_variable_names = [re.sub("_", " ", re.sub(r"\(.*\)", "", v)) for i, v in enumerate(variables) if i not in masked_idxs]
    nb_variables = len(predicted_variable_names)
    generate_time_occurences(series, predicted_variable_names, save, nb_variables, MIN_LENGTH)

    # Visualise Sankey flows
    generate_sankey(series, predicted_variable_names, save, nb_variables, MIN_LENGTH)

    # Visualise series clustering
    generate_clusters(series, save, nb_variables, TAU_MAX+1)

    print(f"Figures saved in results/{save.split('/')[-1]}.")


