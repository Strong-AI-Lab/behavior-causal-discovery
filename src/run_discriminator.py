
import argparse
from typing import Optional, Any, List

from data.dataset import SeriesDataset
from data.structure.chronology import Chronology
from data.structure.loaders import DiscriminatorLoader, DiscriminatorCommunityLoader
from data.constants import TAU_MAX
from model.behaviour_model import TSLinearCausal, DISCRIMINATORS, BEHAVIOUR_MODELS
from script_utils.data_commons import DataManager
from script_utils.graph_commons import load_graph

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


MODELS = {
    **BEHAVIOUR_MODELS,
}

# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('model_save', type=str, help='Load the parametric model or the causal graph from a save folder.')
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
parser.add_argument('--tau_max', type=int, default=TAU_MAX, help='Maximum lag to consider.')
parser.add_argument('--fix_errors_data', action="store_true", help='If specified, fixes simple errors and fills missing values in the data using estimation heuristics.')
parser.add_argument('--filter_null_state_trajectories', action="store_true", help='If specified, removes trajectories with null states from data.')
parser.add_argument('--do_not_skip_stationary', action="store_false", dest="skip_stationary", help='If specified, does not skip stationary trajectories when loading data.')
args = parser.parse_args()

if args.discriminator_save is None and args.train_data_path is None:
    raise ValueError("If a discriminator save is not provided, a training data path must be provided.")

if args.discriminator_type not in DISCRIMINATORS.keys():
    raise ValueError(f"Discriminator type {args.discriminator_type} not supported. Options: {','.join(DISCRIMINATORS.keys())}.")

if args.model_type not in MODELS.keys():
    raise ValueError(f"Model type {args.model_type} not supported. Options: {','.join(MODELS.keys())}.")



# Set variables. /!\ Chronology will be re-written twice if force_data_computation is enabled. User must also make sure that variables between train and test set are identical.
chronology = DataManager.load_data(
    path=args.data_path,
    data_type=Chronology,
    force_data_computation=args.force_data_computation,
    saving_allowed=True,
)
variables = chronology.get_labels()
masked_variables = [var for var in variables if var not in chronology.get_labels('behaviour')]
num_variables = len(variables)

loader = DiscriminatorCommunityLoader if args.community else DiscriminatorLoader


# Create lazy model class
class LazyModel(): # Lazy loading model to avoid loading it if not needed (i.e. if the dataset is loaded from a save and not re-computed)
    def __init__(self, model_type : str, model_savefile : str, filter : Optional[List[str]] = None):
        self.model_type = model_type
        self.model_savefile = model_savefile
        self.filter = [] if filter is None else filter
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
            graph_weights = load_graph(self.model_savefile, variables, self.filter, "all")
            model = TSLinearCausal(num_variables, args.tau_max+1, graph_weights=graph_weights)
            model = model.to(self.device)
        else:
            print("Parametric model detected.")
            model = MODELS[self.model_type].load_from_checkpoint(self.model_savefile, num_variables=num_variables, lookback=args.tau_max+1, map_location=self.device)

        return model

    def __call__(self, *args, **kwargs) -> Any:
        if self.model is None:
            self.model = self.load_model()
        
        return self.model(*args, **kwargs)


# Create optional model
model = LazyModel(args.model_type, args.model_save, None if args.filter is None else args.filter.split(","))


# Read data
test_discr_dataset = DataManager.load_data(
    path=args.data_path,
    data_type=SeriesDataset,
    loader_type=loader,
    dataset_kwargs={"model": model},
    chronology_kwargs={"fix_errors": args.fix_errors_data, "filter_null_state_trajectories": args.filter_null_state_trajectories},
    loader_kwargs={"lookback": args.tau_max+1, "skip_stationary": args.skip_stationary, "vector_columns": variables, "masked_variables": masked_variables},
    force_data_computation=args.force_data_computation,
    saving_allowed=True,
)

if args.discriminator_save is None:
    train_discr_dataset = DataManager.load_data(
        path=args.train_data_path,
        data_type=SeriesDataset,
        loader_type=loader,
        dataset_kwargs={"model": model},
        chronology_kwargs={"fix_errors": args.fix_errors_data, "filter_null_state_trajectories": args.filter_null_state_trajectories},
        loader_kwargs={"lookback": args.tau_max+1, "skip_stationary": args.skip_stationary, "vector_columns": variables, "masked_variables": masked_variables},
        force_data_computation=args.force_data_computation,
        saving_allowed=True,
    )

print(f"Graph with {num_variables} variables: {variables}.")


# Build discriminator
test_loader = DataLoader(test_discr_dataset, batch_size=64, shuffle=True)

if args.discriminator_save is not None:
    print(f"Discriminator save provided. Loading results from {args.discriminator_save}...")
    discriminator = DISCRIMINATORS[args.discriminator_type].load_from_checkpoint(args.discriminator_save, num_variables=num_variables-len(masked_variables), lookback=args.tau_max+1)
else:
    # Train discriminator
    print("Discriminator save not provided. Building a new discriminator.")
    discriminator = DISCRIMINATORS[args.discriminator_type](num_variables-len(masked_variables), args.tau_max+1)
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
trainer.test(discriminator, test_loader)
