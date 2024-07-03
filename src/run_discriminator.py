
import argparse
import os
import re
import numpy as np
import tqdm
from typing import List, Optional

from data.dataset import SeriesDataset
from data.structure.loaders import GeneratorLoader, DiscriminatorLoader, DiscriminatorCommunityLoader
from data.structure.chronology import Chronology
from data.constants import MASKED_VARIABLES, VECTOR_COLUMNS
from model.behaviour_model import TSLinearCausal, DISCRIMINATORS, BEHAVIOUR_MODELS
from model.causal_graph_formatter import CausalGraphFormatter

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


MODELS = {
    **BEHAVIOUR_MODELS,
}

# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('save', type=str, help='Load the parametric model or the causal graph from a save folder.')
parser.add_argument('--discriminator_save', type=str, default=None, help='If provided, loads the discriminator from a save folder instead of running the algorithm again.')
parser.add_argument('--discriminator_type', type=str, default="lstm", help=f'Type of discriminator to use. Options: {",".join(DISCRIMINATORS.keys())}.')
parser.add_argument('--model_type', type=str, default="causal", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser.add_argument('--community', action='store_true', help='If provided, the discriminator is trained and evaluated on community-generated data.')
parser.add_argument('--filter', type=str, default=None, help='If provided and the model is non-parametric, filters the causal graph to only include the most significant links. Options: ' + 
                                                                '"low"  : remove links with low values; ' +
                                                                '"neighbor_effect" : remove links to neighbors, ' + 
                                                                '"corr" : remove correlations without causation. ' +
                                                                'Multiple filters can be applied by separating them with a comma.')
args = parser.parse_args()

print(f"Arguments: save={args.save}")
if args.discriminator_save is not None:
    discriminator_save = args.discriminator_save
    print(f"Arguments: discriminator_save={discriminator_save}")

assert args.discriminator_type in DISCRIMINATORS.keys(), f"Discriminator type {args.discriminator_type} not supported. Options: {','.join(DISCRIMINATORS.keys())}."
assert args.model_type in MODELS.keys(), f"Model type {args.model_type} not supported. Options: {','.join(MODELS.keys())}."


# Set constants
TAU_MAX = 5
LOW_FILTER = 0.075
variables = VECTOR_COLUMNS
num_variables = len(variables)

train_structure_savefile = "train_chronology_behaviours.json"
test_tructure_savefile = "test_chronology_behaviours.json"

if args.community:
    train_dataset_savefile = "train_data_community_discriminator.pt"
    test_dataset_savefile =  "test_data_community_discriminator.pt"
    loader = DiscriminatorCommunityLoader
else:
    train_dataset_savefile = "train_data_discriminator.pt"
    test_dataset_savefile = "test_data_discriminator.pt"
    loader = DiscriminatorLoader


# Load functions
def load_model(model_type : str, model_savefile : str, filter : Optional[str] = None):
    if load_model.model is None:
        # Get model first as required to generate data
        print(f"No dataset found in data/gen. Model save provided. Loading {model_type} model from {model_savefile}...")
        if model_type == "causal":
            print("Causal model detected.")
            val_matrix = np.load(f'{model_savefile}/val_matrix.npy')
            graph = np.load(f'{model_savefile}/graph.npy')

            if filter is not None:
                for f in filter.split(","):
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

            load_model.model = TSLinearCausal(num_variables, TAU_MAX+1, graph_weights=graph*val_matrix)
        else:
            print("Parametric model detected.")
            if torch.cuda.is_available():
                map_location=torch.device('cuda')
            else:
                map_location=torch.device('cpu')
            load_model.model = MODELS[model_type].load_from_checkpoint(model_savefile, num_variables=num_variables, lookback=TAU_MAX+1, map_location=map_location)
    
    return load_model.model
load_model.model = None


def load_data_files(data_files : List[str], structure_savefile : str, dataset_savefile : str, loader : GeneratorLoader, model_type : str, model_savefile : str, filter : Optional[str] = None) -> SeriesDataset:
    if os.path.exists(f"data/gen/{dataset_savefile}"):
        print(f"Loading dataset from data/gen/{dataset_savefile}...")
        dataset = SeriesDataset.load(f"data/gen/{dataset_savefile}")

    else:
        model = load_model(model_type, model_savefile, filter)

        if os.path.exists(f"data/gen/{structure_savefile}"):
            print(f"No dataset found in data/gen. Data structure found and loaded from data/gen/{structure_savefile}. Re-computing the dataset from structure...")

            # Create dataset
            chronology = Chronology.deserialize(f"data/gen/{structure_savefile}")
            structure_loader = loader(lookback=TAU_MAX+1, skip_stationary=True, vector_columns=VECTOR_COLUMNS, masked_variables=MASKED_VARIABLES)
            dataset = SeriesDataset(chronology=chronology, struct_loader=structure_loader, model=model)

            #Save dataset
            dataset.save(f"data/gen/{dataset_savefile}")

        else:
            print("No dataset or data structure found in data/gen. Generating dataset...")

            # Create structure
            chronology = Chronology.create(data_files, fix_errors=True, filter_null_state_trajectories=True)

            # Save structure
            os.makedirs("data/gen", exist_ok=True)
            chronology.serialize(f"data/gen/{structure_savefile}")

            # Create dataset
            structure_loader = loader(lookback=TAU_MAX+1, skip_stationary=True, vector_columns=VECTOR_COLUMNS, masked_variables=MASKED_VARIABLES)
            dataset = SeriesDataset(chronology=chronology, struct_loader=structure_loader, model=model)

            # Save dataset
            dataset.save(f"data/gen/{dataset_savefile}")

    return dataset




# Read data
test_data_files = [f'data/test/{name}' for name in os.listdir('data/test') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
test_discr_dataset = load_data_files(test_data_files, test_tructure_savefile, test_dataset_savefile, loader, args.model_type, args.save, args.filter)

if args.discriminator_save is None:
    train_data_files = [f'data/train/{name}' for name in os.listdir('data/train') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
    train_discr_dataset = load_data_files(train_data_files, train_structure_savefile, train_dataset_savefile, loader, args.model_type, args.save, args.filter)

print(f"Graph with {num_variables} variables: {variables}.")


# Build discriminator
test_loader = DataLoader(test_discr_dataset, batch_size=64, shuffle=True)

if args.discriminator_save is not None:
    print(f"Discriminator save provided. Loading results from {discriminator_save}...")
    discriminator = DISCRIMINATORS[args.discriminator_type].load_from_checkpoint(args.save, num_variables=num_variables-len(MASKED_VARIABLES), lookback=TAU_MAX+1)
else:
    # Train discriminator
    print("Discriminator save not provided. Building a new discriminator.")
    discriminator = DISCRIMINATORS[args.discriminator_type](num_variables-len(MASKED_VARIABLES), TAU_MAX+1)
    discriminator.train()


trainer = pl.Trainer(
        max_epochs=10,
        devices=[0], 
        accelerator="gpu")

if args.discriminator_save is None:
    print("Training discriminator...")
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
