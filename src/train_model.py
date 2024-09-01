
import argparse

from data.dataset import SeriesDataset
from data.structure.chronology import Chronology
from data.structure.loaders import BehaviourSeriesLoader
from data.constants import TAU_MAX
from model.behaviour_model import BEHAVIOUR_MODELS
from script_utils.data_commons import DataManager
from script_utils.graph_commons import load_graph

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


MODELS = {
    **BEHAVIOUR_MODELS,
}


# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str, help='Path to the data folder.')
parser.add_argument('--model_type',type=str, default="lstm", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser.add_argument('--model_save', type=str, default=None, help='If provided, loads the model from a save. The save can be a `model.ckpt` file. If the model_type if `causal_*`, a save folder from a causal_discovery run can also be used.')
parser.add_argument('--wandb_project', type=str, default=None, help='If specified, logs the run to wandb under the specified project.')
parser.add_argument('--force_data_computation', action="store_true", help='If specified, forces the computation of the force data from the raw data.')
parser.add_argument('--causal_graph', type=str, default="all", help='Only used when a save folder from a causal discovery run is loaded. Controls if the graph contains the edges the coefficients. Options: "all", "coefficients", "edges".')
parser.add_argument('--filter', type=str, default=None, help='If provided, filters the causal graph to only include the most significant links. Options: ' + 
                                                                '"low" : remove links with low values; ' +
                                                                '"neighbor_effect" : remove links to neighbors, ' + 
                                                                '"corr" : remove correlations without causation. ' +
                                                                'Multiple filters can be applied by separating them with a comma.')
parser.add_argument('--tau_max', type=int, default=TAU_MAX, help='Maximum lag to consider.')
parser.add_argument('--fix_errors_data', action="store_true", help='If specified, fixes simple errors and fills missing values in the data using estimation heuristics.')
parser.add_argument('--filter_null_state_trajectories', action="store_true", help='If specified, removes trajectories with null states from data.')
parser.add_argument('--do_not_skip_stationary', action="store_false", dest="skip_stationary", help='If specified, does not skip stationary trajectories when loading data.')
args = parser.parse_args()

assert args.model_type in MODELS.keys(), f"Model type {args.model_type} not supported. Options: {','.join(MODELS.keys())}."
assert args.model_type != "causal", f"Model type {args.model_type} does not support training. Use `run_discovery.py` instead."


# Set variables. /!\ Chronology will be re-written twice if force_data_computation is enabled.
chronology = DataManager.load_data(
    path=args.data_path,
    data_type=Chronology,
    force_data_computation=args.force_data_computation,
    saving_allowed=True,
)
variables = chronology.get_labels()
masked_variables = [var for var in variables if var not in chronology.get_labels('behaviour')]
num_variables = len(variables)

print(f"Graph with {num_variables} variables: {variables}.")

# Load dataset
train_dataset = DataManager.load_data(
    path=args.data_path,
    data_type=SeriesDataset,
    loader_type=BehaviourSeriesLoader,
    chronology_kwargs={"fix_errors": args.fix_errors_data, "filter_null_state_trajectories": args.filter_null_state_trajectories},
    loader_kwargs={"lookback": args.tau_max+1, "skip_stationary": args.skip_stationary, "vector_columns": variables},
    force_data_computation=args.force_data_computation,
    saving_allowed=True,
)


# Build data loader
dataset = torch.utils.data.random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset))], generator=torch.Generator().manual_seed(1))
train_loader = DataLoader(dataset[0], batch_size=64, shuffle=True)
val_loader = DataLoader(dataset[1], batch_size=64, shuffle=False)

# Mask context variables for predition
masked_idxs = [variables.index(var) for var in masked_variables]
print(f"Masking {len(masked_idxs)} variables: {masked_variables}")


# Build model
if args.model_save is None:
        model = MODELS[args.model_type](num_variables=num_variables, lookback=args.tau_max+1, masked_idxs_for_training=masked_idxs)
else:
    print(f"Save provided. Loading {args.model_type} model from {args.model_save}...")
    if args.model_type.startswith("causal_") and not args.model_save.endswith(".ckpt"):
        print("Causal model detected.")
        graph_weights = load_graph(args.model_save, variables, [] if args.filter is None else args.filter.split(","), args.causal_graph)
        model = MODELS[args.model_type](num_variables=num_variables, lookback=args.tau_max+1, graph_weights=graph_weights, masked_idxs_for_training=masked_idxs)
    else:
        print("Parametric model detected.")
        model = MODELS[args.model_type].load_from_checkpoint(args.model_save, num_variables=num_variables, lookback=args.tau_max+1, masked_idxs_for_training=masked_idxs)


# Train model
trainer = pl.Trainer(
    max_epochs=10,
    devices=[0], 
    accelerator="gpu",
    logger=WandbLogger(name=f"{args.model_type}_train", project=args.wandb_project) if args.wandb_project else None,
    )

trainer.fit(model, train_loader, val_loader)