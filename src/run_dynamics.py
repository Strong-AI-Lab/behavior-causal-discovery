
import argparse

from data.dataset import SeriesDataset
from data.structure.loaders import DynamicSeriesLoader, DynamicGraphSeriesLoader
from model.dynamics_model import DYNAMIC_MODELS
from model.graph_dynamics_model import GRAPH_DYNAMIC_MODELS
from script_utils.data_commons import DataManager
from script_utils.parser_commons import add_loader_arguments_to_parser, add_lookback_arguments_to_parser

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
parser.add_argument('model_save', type=str, help='Load the model from a save folder.')
parser.add_argument('data_path', type=str, help='Path to the data folder.')
parser.add_argument('--model_type',type=str, default="dynamical_lstm", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser.add_argument('--wandb_project', type=str, default=None, help='If specified, logs the run to wandb under the specified project.')
parser = add_lookback_arguments_to_parser(parser)
parser = add_loader_arguments_to_parser(parser)
args = parser.parse_args()

assert args.model_type in MODELS.keys(), f"Model type {args.model_type} not supported. Options: {','.join(MODELS.keys())}."

is_graph_model = False
if args.model_type in GRAPH_DYNAMIC_MODELS.keys():
    is_graph_model = True


# Load dataset
test_dataset = DataManager.load_data(
    path=args.data_path,
    data_type=SeriesDataset,
    loader_type=DynamicGraphSeriesLoader if is_graph_model else DynamicSeriesLoader,
    chronology_kwargs={"fix_errors": args.fix_errors_data, "filter_null_state_trajectories": args.filter_null_state_trajectories},
    loader_kwargs={"lookback": args.tau_max+1, "skip_stationary": args.skip_stationary},
    force_data_computation=args.force_data_computation,
    saving_allowed=True,
)


# Build data loader
if is_graph_model:
    loader = GraphDataLoader(test_dataset, batch_size=64, shuffle=False)
else:
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



# Build model
model = MODELS[args.model_type].load_from_checkpoint(args.model_save, lookback=args.tau_max+1)


# Test model
trainer = pl.Trainer(
    max_epochs=20,
    devices=[0],
    accelerator="gpu",
    logger=WandbLogger(name=f"{args.model_type}_test", project=args.wandb_project) if args.wandb_project else None,
)

trainer.test(model, loader)