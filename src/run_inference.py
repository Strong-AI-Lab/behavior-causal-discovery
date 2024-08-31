
import argparse
import os
import re
import numpy as np

from data.dataset import SeriesDataset
from data.structure.chronology import Chronology
from data.structure.loaders import BehaviourSeriesLoader, GeneratorLoader, GeneratorCommunityLoader
from data.constants import VECTOR_COLUMNS, MASKED_VARIABLES
from model.behaviour_model import TSLinearCausal, BEHAVIOUR_MODELS
from model.causal_graph_formatter import CausalGraphFormatter
from evaluate.evaluation import direct_prediction_accuracy, mutual_information
from evaluate.visualisation import generate_time_occurences, generate_sankey, generate_clusters
from script_utils.data_commons import DataManager

import torch
from torch.utils.data import DataLoader


MODELS = {
    **BEHAVIOUR_MODELS,
}


# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('save', type=str, help='Load the causal graph from a save folder.')
parser.add_argument('data_path', type=str, help='Path to the data folder.')
parser.add_argument('--model_type', type=str, default="causal", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser.add_argument('--filter', type=str, default=None, help='If provided, filters the causal graph to only include the most significant links. Options: ' + 
                                                                '"low"  : remove links with low values; ' +
                                                                '"neighbor_effect" : remove links to neighbors, ' + 
                                                                '"corr" : remove correlations without causation. ' +
                                                                'Multiple filters can be applied by separating them with a comma.')
parser.add_argument('--force_data_computation', action="store_true", help='If specified, forces the computation of the force data from the raw data.')

args = parser.parse_args()

save = args.save
print(f"Arguments: save={save}")
if save.endswith('/'):
    save = save[:-1]


# Set constants
TAU_MAX = 5
LOW_FILTER = 0.075


# Load chronology and dataset
test_dataset = DataManager.load_data(
    path=args.data_path,
    data_type=SeriesDataset,
    loader_type=BehaviourSeriesLoader,
    chronology_kwargs={"fix_errors": True, "filter_null_state_trajectories": True},
    loader_kwargs={"lookback": TAU_MAX+1, "skip_stationary": True},
    force_data_computation=args.force_data_computation,
    saving_allowed=True,
)
chronology = DataManager.load_data(
    path=args.data_path,
    data_type=Chronology,
    force_data_computation=False,
    saving_allowed=False,
)



variables = VECTOR_COLUMNS
num_variables = len(variables)
print(f"Graph with {num_variables} variables: {variables}.")

random_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)


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

    model = TSLinearCausal(num_variables, TAU_MAX+1, graph_weights=graph*val_matrix)
else:
    print("Parametric model detected.")
    if torch.cuda.is_available():
        map_location=torch.device('cuda')
    else:
        map_location=torch.device('cpu')
    model = MODELS[args.model_type].load_from_checkpoint(save, num_variables=num_variables, lookback=TAU_MAX+1, map_location=map_location)

    save_split = save.split('/')
    save = "/".join(save_split[:-3] + [save_split[-3] + "_" + save_split[-1][:-5]]) # Remove .ckpt from save path and concact with version to build results directory



# Mask context variables for predition
MIN_LENGTH = 30
masked_idxs = [variables.index(var) for var in MASKED_VARIABLES]
close_neighbor_idxs = [variables.index(var) for var in variables if var.startswith('close_neighbour_') and not var.endswith('_zone')]
distant_neighbor_idxs = [variables.index(var) for var in variables if var.startswith('distant_neighbour_') and not var.endswith('_zone')]
print(f"Masking {len(masked_idxs)} variables: {MASKED_VARIABLES}")



# Evaluate model
model.eval()
with torch.no_grad():
    # Compute direct prediction accuracy
    acc, acc_last = direct_prediction_accuracy(model, random_loader, num_variables, masked_idxs)
    print(f"Direct Prediction Accuracy: {acc}")
    print(f"Direct Prediction Accuracy (last layer only): {acc_last}")

    # Compute conditional mutual information
    cmi = mutual_information(model, random_loader, num_variables, masked_idxs)
    print(f"Mutual Information: {cmi}")


    # Generate series
    generation_loader = GeneratorLoader(TAU_MAX+1, skip_stationary=True, vector_columns=VECTOR_COLUMNS, masked_variables=MASKED_VARIABLES)
    series = generation_loader.load(chronology, model, build_series=True)

    # Visualise time occurences for series
    predicted_variable_names = [re.sub("_", " ", re.sub(r"\(.*\)", "", v)) for i, v in enumerate(variables) if i not in masked_idxs]
    nb_variables = len(predicted_variable_names)
    generate_time_occurences(series, predicted_variable_names, save, nb_variables, MIN_LENGTH)

    # Visualise Sankey flows
    generate_sankey(series, predicted_variable_names, save, nb_variables, MIN_LENGTH)

    # Visualise series clustering
    generate_clusters(series, save, nb_variables)


    # Generate community series
    community_generation_loader = GeneratorCommunityLoader(TAU_MAX+1, skip_stationary=True, vector_columns=VECTOR_COLUMNS, masked_variables=MASKED_VARIABLES)
    community_series = community_generation_loader.load(chronology, model, build_series=True)

    # Visualise time occurences for community series
    generate_time_occurences(community_series, predicted_variable_names, save, nb_variables, MIN_LENGTH, prefix="community")

    # Visualise Sankey flows
    generate_sankey(community_series, predicted_variable_names, save, nb_variables, MIN_LENGTH, prefix="community")

    # Visualise series clustering
    generate_clusters(community_series, save, nb_variables, prefix="community")

    print(f"Figures saved in results/{save.split('/')[-1]}.")


