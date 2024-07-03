
import argparse
import os
import re
import numpy as np
import pickle

from data.dataset import SeriesDataset
from data.structure.chronology import Chronology
from data.structure.loaders import BehaviourSeriesLoader, GeneratorLoader, GeneratorCommunityLoader
from data.constants import VECTOR_COLUMNS, MASKED_VARIABLES
from model.behaviour_model import TSLinearCausal, BEHAVIOUR_MODELS
from model.causal_graph_formatter import CausalGraphFormatter
from evaluate.evaluation import direct_prediction_accuracy, mutual_information
from evaluate.visualisation import generate_time_occurences, generate_sankey, generate_clusters

import torch
from torch.utils.data import DataLoader


MODELS = {
    **BEHAVIOUR_MODELS,
}


# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('save', type=str, help='Load the causal graph from a save folder.')
parser.add_argument('--model_type', type=str, default="causal", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser.add_argument('--filter', type=str, default=None, help='If provided, filters the causal graph to only include the most significant links. Options: ' + 
                                                                '"low"  : remove links with low values; ' +
                                                                '"neighbor_effect" : remove links to neighbors, ' + 
                                                                '"corr" : remove correlations without causation. ' +
                                                                'Multiple filters can be applied by separating them with a comma.')
args = parser.parse_args()

save = args.save
print(f"Arguments: save={save}")
if save.endswith('/'):
    save = save[:-1]


# Set constants
TAU_MAX = 5
LOW_FILTER = 0.075


structure_savefile = "test_chronology_behaviours.json"
dataset_savefile = "test_data_behaviour.pt"

generation_savefile = "test_data_generation"
i = 0
while os.path.exists(f"data/gen/{generation_savefile}_{i}.pickle"): # File is "test_data_generation_<i>.pickle", increment i if file already exists
    i += 1
generation_savefile = f"{generation_savefile}_{i}.pickle"

community_generation_savefile = "test_data_community_generation"
i = 0
while os.path.exists(f"data/gen/{community_generation_savefile}_{i}.pickle"): # File is "test_data_community_generation_<i>.pickle", increment i if file already exists
    i += 1
community_generation_savefile = f"{community_generation_savefile}_{i}.pickle"


if os.path.exists(f"data/gen/{structure_savefile}"):
    print(f"Data structure found and loaded from data/gen/{structure_savefile}.")

    # Create structure
    chronology = Chronology.deserialize(f"data/gen/{structure_savefile}")

    # Create or load dataset
    if not os.path.exists(f"data/gen/{dataset_savefile}"):
        behaviour_loader = BehaviourSeriesLoader(TAU_MAX+1, skip_stationary=True)
        test_dataset = SeriesDataset(chronology=chronology, struct_loader=behaviour_loader)
        test_dataset.save(f"data/gen/{dataset_savefile}")
    else:
        test_dataset = SeriesDataset.load(f"data/gen/{dataset_savefile}")

else:
    if args.force_data_computation:
        print("Forced data computation. (Re)computing dataset...")
    else:
        print("No dataset or data structure found in data/gen. Generating dataset...")

    # Create structure and dataset
    chronology = Chronology.create([f'data/test/{name}' for name in os.listdir('data/train') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)])

    structure_loader = BehaviourSeriesLoader(lookback=TAU_MAX+1, skip_stationary=True)
    test_dataset = SeriesDataset(chronology=chronology, struct_loader=structure_loader)

    # Save structure
    os.makedirs("data/gen", exist_ok=True)
    chronology.serialize(f"data/gen/{structure_savefile}")

    # Save datasets
    test_dataset.save(f"data/gen/{dataset_savefile}")



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
    generation_loader = GeneratorLoader(TAU_MAX+1, skip_stationary=True, vector_columns=VECTOR_COLUMNS, masked_columns=MASKED_VARIABLES)
    series = generation_loader.load(chronology, model, build_series=True)

    # Save series
    with open(f"data/gen/{generation_savefile}", 'wb') as f:
        pickle.dump(series, f)

    # Visualise time occurences for series
    predicted_variable_names = [re.sub("_", " ", re.sub(r"\(.*\)", "", v)) for i, v in enumerate(variables) if i not in masked_idxs]
    nb_variables = len(predicted_variable_names)
    generate_time_occurences(series, predicted_variable_names, save, nb_variables, MIN_LENGTH)

    # Visualise Sankey flows
    generate_sankey(series, predicted_variable_names, save, nb_variables, MIN_LENGTH)

    # Visualise series clustering
    generate_clusters(series, save, nb_variables)


    # Generate community series
    community_generation_loader = GeneratorCommunityLoader(TAU_MAX+1, skip_stationary=True, vector_columns=VECTOR_COLUMNS, masked_columns=MASKED_VARIABLES)
    community_series = community_generation_loader.load(chronology, model, build_series=True)

    # Save community series
    with open(f"data/gen/{community_generation_savefile}", 'wb') as f:
        pickle.dump(community_series, f)

    # Visualise time occurences for community series
    generate_time_occurences(community_series, predicted_variable_names, save, nb_variables, MIN_LENGTH, prefix="community")

    # Visualise Sankey flows
    generate_sankey(community_series, predicted_variable_names, save, nb_variables, MIN_LENGTH, prefix="community")

    # Visualise series clustering
    generate_clusters(community_series, save, nb_variables, prefix="community")

    print(f"Figures saved in results/{save.split('/')[-1]}.")


