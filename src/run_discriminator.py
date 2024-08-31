
import argparse
import numpy as np
import tqdm
from typing import Optional, Any

from data.dataset import SeriesDataset
from data.structure.loaders import DiscriminatorLoader, DiscriminatorCommunityLoader
from data.constants import MASKED_VARIABLES, VECTOR_COLUMNS
from model.behaviour_model import TSLinearCausal, DISCRIMINATORS, BEHAVIOUR_MODELS
from model.causal_graph_formatter import CausalGraphFormatter
from script_utils.data_commons import DataManager

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
parser.add_argument('data_path', type=str, help='Path to the data folder.')
parser.add_argument('--discriminator_save', type=str, default=None, help='If provided, loads the discriminator from a save folder instead of running the algorithm again.')
parser.add_argument('--train_data_path', type=str, help='Path to the training data folder. Required if a discriminator save is not provided to train the discriminator.')
parser.add_argument('--discriminator_type', type=str, default="lstm", help=f'Type of discriminator to use. Options: {",".join(DISCRIMINATORS.keys())}.')
parser.add_argument('--model_type', type=str, default="causal", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser.add_argument('--community', action='store_true', help='If provided, the discriminator is trained and evaluated on community-generated data.')
parser.add_argument('--filter', type=str, default=None, help='If provided and the model is non-parametric, filters the causal graph to only include the most significant links. Options: ' + 
                                                                '"low"  : remove links with low values; ' +
                                                                '"neighbor_effect" : remove links to neighbors, ' + 
                                                                '"corr" : remove correlations without causation. ' +
                                                                'Multiple filters can be applied by separating them with a comma.')
parser.add_argument('--force_data_computation', action="store_true", help='If specified, forces the computation of the force data from the raw data.')
args = parser.parse_args()

print(f"Arguments: save={args.save}")
if args.discriminator_save is not None:
    discriminator_save = args.discriminator_save
    print(f"Arguments: discriminator_save={discriminator_save}")
elif args.train_data_path is None:
    raise ValueError("If a discriminator save is not provided, a training data path must be provided.")

if args.discriminator_type not in DISCRIMINATORS.keys():
    raise ValueError(f"Discriminator type {args.discriminator_type} not supported. Options: {','.join(DISCRIMINATORS.keys())}.")

if args.model_type not in MODELS.keys():
    raise ValueError(f"Model type {args.model_type} not supported. Options: {','.join(MODELS.keys())}.")


# Set constants
TAU_MAX = 5
LOW_FILTER = 0.075
variables = VECTOR_COLUMNS
num_variables = len(variables)

loader = DiscriminatorCommunityLoader if args.community else DiscriminatorLoader


# Create lazy model class
class LazyModel(): # Lazy loading model to avoid loading it if not needed (i.e. if the dataset is loaded from a save and not re-computed)
    def __init__(self, model_type : str, model_savefile : str, filter : Optional[str] = None):
        self.model_type = model_type
        self.model_savefile = model_savefile
        self.filter = filter
        self.model = None
        
        if torch.cuda.is_available():
            map_location=torch.device('cuda')
        else:
            map_location=torch.device('cpu')
        self.device = map_location

    def load_model(self):
        print(f"Model save provided. Loading {self.model_type} model from {self.model_savefile}...")
        if self.model_type == "causal":
            print("Causal model detected.")
            val_matrix = np.load(f'{self.model_savefile}/val_matrix.npy')
            graph = np.load(f'{self.model_savefile}/graph.npy')

            if self.filter is not None:
                for f in self.filter.split(","):
                    print(f"Filtering results using {f}...")
                    if f == 'low':
                        filtered_values = CausalGraphFormatter(graph, val_matrix).low_filter(LOW_FILTER)
                        val_matrix = filtered_values.get_val_matrix()
                        graph = filtered_values.get_graph()
                    elif f == "neighbor_effect":
                        filtered_values = CausalGraphFormatter(graph, val_matrix).var_filter([], [variables.index(v) for v in variables if v.startswith('close_neighbour_') or v.startswith('distant_neighbour_')])
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
            model = MODELS[self.model_type].load_from_checkpoint(self.model_savefile, num_variables=num_variables, lookback=TAU_MAX+1, map_location=self.device)

        return model

    def __call__(self, *args, **kwargs) -> Any:
        if self.model is None:
            self.model = self.load_model()
        
        return self.model(*args, **kwargs)


# Create optional model
model = LazyModel(args.model_type, args.save, args.filter)



# Read data
test_discr_dataset = DataManager.load_data(
    path=args.data_path,
    data_type=SeriesDataset,
    loader_type=loader,
    dataset_kwargs={"model": model},
    loader_kwargs={"lookback": TAU_MAX+1, "skip_stationary": True, "vector_columns": VECTOR_COLUMNS, "masked_variables": MASKED_VARIABLES},
    force_data_computation=args.force_data_computation,
    saving_allowed=True,
)

if args.discriminator_save is None:
    train_discr_dataset = DataManager.load_data(
        path=args.train_data_path,
        data_type=SeriesDataset,
        loader_type=loader,
        dataset_kwargs={"model": model},
        loader_kwargs={"lookback": TAU_MAX+1, "skip_stationary": True, "vector_columns": VECTOR_COLUMNS, "masked_variables": MASKED_VARIABLES},
        force_data_computation=args.force_data_computation,
        saving_allowed=True,
    )

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
