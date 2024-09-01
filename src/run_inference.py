
import argparse
import re
import os

from data.dataset import SeriesDataset
from data.structure.chronology import Chronology
from data.structure.loaders import BehaviourSeriesLoader, GeneratorLoader, GeneratorCommunityLoader
from data.constants import VECTOR_COLUMNS, MASKED_VARIABLES, TAU_MAX, RESULTS_SAVE_FOLDER_DEFAULT
from model.behaviour_model import TSLinearCausal, BEHAVIOUR_MODELS
from evaluate.evaluation import direct_prediction_accuracy, mutual_information
from evaluate.visualisation import generate_time_occurences, generate_sankey, generate_clusters
from script_utils.data_commons import DataManager
from script_utils.graph_commons import load_graph

import torch
from torch.utils.data import DataLoader


MODELS = {
    **BEHAVIOUR_MODELS,
}


# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('model_save', type=str, help='Load the causal graph from a save folder.')
parser.add_argument('data_path', type=str, help='Path to the data folder.')
parser.add_argument('--save_folder', type=str, default=RESULTS_SAVE_FOLDER_DEFAULT, help='Folder to save the results.')
parser.add_argument('--model_type', type=str, default="causal", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser.add_argument('--filter', type=str, default=None, help='If provided, filters the causal graph to only include the most significant links. Options: ' + 
                                                                '"low"  : remove links with low values; ' +
                                                                '"neighbor_effect" : remove links to neighbors, ' + 
                                                                '"corr" : remove correlations without causation. ' +
                                                                'Multiple filters can be applied by separating them with a comma.')
parser.add_argument('--force_data_computation', action="store_true", help='If specified, forces the computation of the force data from the raw data.')
parser.add_argument('--tau_max', type=int, default=TAU_MAX, help='Maximum lag to consider.')
parser.add_argument('--fix_errors_data', action="store_true", help='If specified, fixes simple errors and fills missing values in the data using estimation heuristics.')
parser.add_argument('--filter_null_state_trajectories', action="store_true", help='If specified, removes trajectories with null states from data.')
parser.add_argument('--do_not_skip_stationary', action="store_false", dest="skip_stationary", help='If specified, does not skip stationary trajectories when loading data.')
args = parser.parse_args()

model_save = os.path.normpath(args.model_save)


# Load chronology and dataset
test_dataset = DataManager.load_data(
    path=args.data_path,
    data_type=SeriesDataset,
    loader_type=BehaviourSeriesLoader,
    chronology_kwargs={"fix_errors": args.fix_errors_data, "filter_null_state_trajectories": args.filter_null_state_trajectories},
    loader_kwargs={"lookback": args.tau_max+1, "skip_stationary": args.skip_stationary},
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
print(f"Save provided. Loading {args.model_type} model from {model_save}...")
if torch.cuda.is_available():
    map_location=torch.device('cuda')
else:
    map_location=torch.device('cpu')

if args.model_type == "causal":
    print("Causal model detected.")
    graph_weights = load_graph(model_save, variables, [] if args.filter is None else args.filter.split(","), "all")
    model = TSLinearCausal(num_variables, args.tau_max+1, graph_weights=graph_weights)
    model = model.to(map_location)
    model_save_name = os.path.basename(model_save)
else:
    print("Parametric model detected.")
    model = MODELS[args.model_type].load_from_checkpoint(model_save, num_variables=num_variables, lookback=args.tau_max+1, map_location=map_location)
    model_path_split = model_save.split(os.sep)
    model_save_name = model_path_split[-2] + "_" + model_path_split[-1][:-5] # Remove .ckpt from save path and concat with version to build results directory

save_folder = os.path.join(args.save_folder, model_save_name)
os.makedirs(save_folder, exist_ok=True)


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


    print(f"Results will be saved in {save_folder}.") 

    # Generate series
    generation_loader = GeneratorLoader(args.tau_max+1, skip_stationary=True, vector_columns=VECTOR_COLUMNS, masked_variables=MASKED_VARIABLES)
    series = generation_loader.load(chronology, model, build_series=True)

    # Visualise time occurences for series
    predicted_variable_names = [re.sub("_", " ", re.sub(r"\(.*\)", "", v)) for i, v in enumerate(variables) if i not in masked_idxs]
    nb_variables = len(predicted_variable_names)
    generate_time_occurences(series, predicted_variable_names, save_folder, nb_variables, MIN_LENGTH)

    # Visualise Sankey flows
    generate_sankey(series, predicted_variable_names, save_folder, nb_variables, MIN_LENGTH)

    # Visualise series clustering
    generate_clusters(series, save_folder, nb_variables)


    # Generate community series
    community_generation_loader = GeneratorCommunityLoader(args.tau_max+1, skip_stationary=True, vector_columns=VECTOR_COLUMNS, masked_variables=MASKED_VARIABLES)
    community_series = community_generation_loader.load(chronology, model, build_series=True)

    # Visualise time occurences for community series
    generate_time_occurences(community_series, predicted_variable_names, save_folder, nb_variables, MIN_LENGTH, prefix="community")

    # Visualise Sankey flows
    generate_sankey(community_series, predicted_variable_names, save_folder, nb_variables, MIN_LENGTH, prefix="community")

    # Visualise series clustering
    generate_clusters(community_series, save_folder, nb_variables, prefix="community")


