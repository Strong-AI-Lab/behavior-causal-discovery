
import argparse

from data.dataset import SeriesDataset
from data.structure.loaders import DynamicSeriesLoader, DynamicGraphSeriesLoader
from model.dynamics_model import DYNAMIC_MODELS
from model.graph_dynamics_model import GRAPH_DYNAMIC_MODELS
from script_utils.data_commons import DataManager
from script_utils.parser_commons import add_loader_arguments_to_parser, add_lookback_arguments_to_parser

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
parser.add_argument('--model_type',type=str, default="dynamical_lstm", choices=MODELS.keys(), help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser.add_argument('--model_save', type=str, default=None, help='If provided, loads the model from a save. The save can be a `model.ckpt` file. If the model_type if `causal_*`, a save folder from a causal_discovery run can also be used.')
parser.add_argument('--wandb_project', type=str, default=None, help='If specified, logs the run to wandb under the specified project.')
parser = add_lookback_arguments_to_parser(parser)
parser = add_loader_arguments_to_parser(parser)
args, unknown_args = parser.parse_known_args()

is_graph_model = False
if args.model_type in GRAPH_DYNAMIC_MODELS.keys():
    is_graph_model = True

model_kwargs = {} # Add model specific arguments from command line
if len(unknown_args) > 0:
    model_parser = MODELS[args.model_type].add_to_parser(argparse.ArgumentParser(add_help=False))
    model_args, unknown_remains = model_parser.parse_known_args(unknown_args)

    if len(unknown_remains) > 0: # Raise error if unknown arguments are provided and exit
        argparse.ArgumentParser(parents=[parser,model_parser],add_help=False).error(f"unrecognized arguments for main parser or {args.model_type} model parser: {' '.join(unknown_remains)}")

    model_kwargs = vars(model_args)


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
    model = MODELS[args.model_type](lookback=args.tau_max+1, **model_kwargs)
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