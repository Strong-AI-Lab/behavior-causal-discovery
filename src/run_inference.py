
import argparse
import re
import os

from data.dataset import SeriesDataset
from data.structure.chronology import Chronology
from data.structure.loaders import BehaviourSeriesLoader, GeneratorLoader, GeneratorCommunityLoader
from data.constants import RESULTS_SAVE_FOLDER_DEFAULT, MAX_CLUSTER_DATA_POINTS
from model.behaviour_model import TSLinearCausal, BEHAVIOUR_MODELS
from evaluate.evaluation import direct_prediction_accuracy, mutual_information
from evaluate.visualisation import generate_time_occurences, generate_sankey, generate_clusters
from script_utils.data_commons import DataManager
from script_utils.graph_commons import load_graph
from script_utils.parser_commons import add_loader_arguments_to_parser, add_lookback_arguments_to_parser, add_causal_arguments_to_parser

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
parser.add_argument('--model_type', type=str, default="causal", choices=MODELS.keys(), help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser = add_causal_arguments_to_parser(parser)
parser = add_lookback_arguments_to_parser(parser)
parser = add_loader_arguments_to_parser(parser)
args = parser.parse_args()
model_save = os.path.normpath(args.model_save)


# Set variables. /!\ Chronology will be re-written twice if force_data_computation is enabled.
chronology = DataManager.load_data(
    path=args.data_path,
    data_type=Chronology,
    chronology_kwargs={"fix_errors": args.fix_errors_data, "filter_null_state_trajectories": args.filter_null_state_trajectories},
    force_data_computation=args.force_data_computation,
    saving_allowed=True,
)
variables = chronology.get_labels()
masked_variables = [var for var in variables if var not in chronology.get_labels('behaviour')]
num_variables = len(variables)

print(f"Graph with {num_variables} variables: {variables}.")


# Load dataset
test_dataset = DataManager.load_data(
    path=args.data_path,
    data_type=SeriesDataset,
    loader_type=BehaviourSeriesLoader,
    chronology_kwargs={"fix_errors": args.fix_errors_data, "filter_null_state_trajectories": args.filter_null_state_trajectories},
    loader_kwargs={"lookback": args.tau_max+1, "skip_stationary": args.skip_stationary, "vector_columns": variables},
    force_data_computation=args.force_data_computation,
    saving_allowed=True,
)

random_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


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
    model_save_name = "_".join(model_path_split[-3:-1]) + "_" + model_path_split[-1][:-5] # Remove .ckpt from save path and concat with version to build results directory

save_folder = os.path.join(args.save_folder, model_save_name)
os.makedirs(save_folder, exist_ok=True)


# Mask context variables for predition
MIN_LENGTH = 30
masked_idxs = [variables.index(var) for var in masked_variables]
print(f"Masking {len(masked_idxs)} variables: {masked_variables}")



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
    generation_loader = GeneratorLoader(args.tau_max+1, skip_stationary=args.skip_stationary, vector_columns=variables, masked_variables=masked_variables)
    series = generation_loader.load(chronology, model, build_series=True)

    # Visualise time occurences for series
    predicted_variable_names = [re.sub("_", " ", re.sub(r"\(.*\)", "", v)) for i, v in enumerate(variables) if i not in masked_idxs]
    nb_variables = len(predicted_variable_names)
    generate_time_occurences(series, predicted_variable_names, save_folder, nb_variables, MIN_LENGTH)
    print(f"Time occurences visualisation saved.")

    # Visualise Sankey flows
    generate_sankey(series, predicted_variable_names, save_folder, nb_variables, MIN_LENGTH)
    print(f"Sankey visualisation saved.")

    # Visualise series clustering
    data_pred, data_truth, _ = generation_loader.load(chronology, model, build_series=False)
    generate_clusters(data_pred, data_truth, save_folder, nb_variables, args.tau_max+1, max_data_points=MAX_CLUSTER_DATA_POINTS)
    print(f"Clusters visualisation saved.")


    # Generate community series
    community_generation_loader = GeneratorCommunityLoader(args.tau_max+1, skip_stationary=args.skip_stationary, vector_columns=variables, masked_variables=masked_variables)
    community_series = community_generation_loader.load(chronology, model, build_series=True)

    # Visualise time occurences for community series
    generate_time_occurences(community_series, predicted_variable_names, save_folder, nb_variables, MIN_LENGTH, prefix="community")
    print(f"Time occurences visualisation saved.")

    # Visualise Sankey flows
    generate_sankey(community_series, predicted_variable_names, save_folder, nb_variables, MIN_LENGTH, prefix="community")
    print(f"Sankey visualisation saved.")

    # Visualise series clustering
    community_data_pred, community_data_truth, _ = community_generation_loader.load(chronology, model, build_series=False)
    generate_clusters(community_data_pred, community_data_truth, save_folder, nb_variables, args.tau_max+1, prefix="community", max_data_points=MAX_CLUSTER_DATA_POINTS)
    print(f"Clusters visualisation saved.")


