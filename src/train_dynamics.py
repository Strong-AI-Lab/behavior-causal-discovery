
import argparse

from data.dataset import SeriesDataset
from data.structure.loaders import DynamicSeriesLoader, DynamicGraphSeriesLoader
from model.dynamics_model import DYNAMIC_MODELS
from model.graph_dynamics_model import GRAPH_DYNAMIC_MODELS
from data.constants import TAU_MAX
from script_utils.data_commons import DataManager

import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


MODELS = {
    **DYNAMIC_MODELS,
    **GRAPH_DYNAMIC_MODELS
}


# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str, help='Path to the data folder.')
parser.add_argument('--model_type',type=str, default="dynamical_lstm", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser.add_argument('--model_save', type=str, default=None, help='If provided, loads the model from a save. The save can be a `model.ckpt` file. If the model_type if `causal_*`, a save folder from a causal_discovery run can also be used.')
parser.add_argument('--wandb_project', type=str, default=None, help='If specified, logs the run to wandb under the specified project.')
parser.add_argument('--force_data_computation', action="store_true", help='If specified, forces the computation of the force data from the raw data.')
parser.add_argument('--tau_max', type=int, default=TAU_MAX, help='Maximum lag to consider.')
parser.add_argument('--fix_errors_data', action="store_true", help='If specified, fixes simple errors and fills missing values in the data using estimation heuristics.')
parser.add_argument('--filter_null_state_trajectories', action="store_true", help='If specified, removes trajectories with null states from data.')
parser.add_argument('--do_not_skip_stationary', action="store_false", dest="skip_stationary", help='If specified, does not skip stationary trajectories when loading data.')
args = parser.parse_args()

assert args.model_type in MODELS.keys(), f"Model type {args.model_type} not supported. Options: {','.join(MODELS.keys())}."

is_graph_model = False
if args.model_type in GRAPH_DYNAMIC_MODELS.keys():
    is_graph_model = True


# Load dataset
train_dataset = DataManager.load_data(
    path=args.data_path,
    data_type=SeriesDataset,
    loader_type=DynamicGraphSeriesLoader if is_graph_model else DynamicSeriesLoader,
    chronology_kwargs={"fix_errors": args.fix_errors_data, "filter_null_state_trajectories": args.filter_null_state_trajectories},
    loader_kwargs={"lookback": args.tau_max+1, "skip_stationary": args.skip_stationary},
    force_data_computation=args.force_data_computation,
    saving_allowed=True,
)


# Build data loader
dataset = torch.utils.data.random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset))], generator=torch.Generator().manual_seed(1))
if is_graph_model:
    train_loader = GraphDataLoader(dataset[0], batch_size=64, shuffle=True)
    val_loader = GraphDataLoader(dataset[1], batch_size=64, shuffle=False)
else:
    train_loader = DataLoader(dataset[0], batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset[1], batch_size=64, shuffle=False)



# Build model
if args.model_save is None:
    print(f"No save provided. Building {args.model_type} model...")
    model = MODELS[args.model_type](lookback=args.tau_max+1)
else:
    print(f"Save provided. Loading {args.model_type} model from {args.model_save}...")
    model = MODELS[args.model_type].load_from_checkpoint(args.model_save, lookback=args.tau_max+1)


# Train model
trainer = pl.Trainer(
    max_epochs=20,
    devices=[0],
    accelerator="gpu",
    logger=WandbLogger(name=f"{args.model_type}_train", project=args.wandb_project) if args.wandb_project else None,
)

trainer.fit(model, train_loader, val_loader)