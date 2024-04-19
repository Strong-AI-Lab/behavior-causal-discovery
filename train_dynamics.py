
import argparse
import os
import re
import pandas as pd
import numpy as np

from src.data.format_data import PandasFormatterEnsemble
from src.data.dataset import DynamicSeriesDataset
from src.model.dynamics_model import DYNAMIC_MODELS
from src.dynamics.solver import DynamicsSolver

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


MODELS = {
    **DYNAMIC_MODELS,
}


# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('--model_type',type=str, default="dynamical_lstm", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser.add_argument('--save', type=str, default=None, help='If provided, loads the model from a save. The save can be a `model.ckpt` file. If the model_type if `causal_*`, a save folder from a causal_discovery run can als be used.')
parser.add_argument('--wandb_project', type=str, default=None, help='If specified, logs the run to wandb under the specified project.')
parser.add_argument('--force_data_computation', action="store_true", help='If specified, forces the computation of the force data from the raw data.')
args = parser.parse_args()

assert args.model_type in MODELS.keys(), f"Model type {args.model_type} not supported. Options: {','.join(MODELS.keys())}."


# Set constants
TAU_MAX = 5
LOW_FILTER = 0.075


if not args.force_data_computation and os.path.exists(f"data/gen/data_dynamics.pt"):
        print("Loading dataset from data/gen/data_dynamics.pt...")
        train_dataset = torch.load("data/gen/data_dynamics.pt")

else:
        if not os.path.exists("data/gen/data_dynamics.pt"):
                print("No dataset found in data/gen. Generating dataset...")
        if args.force_data_computation:
                print("Forced data computation. (Re)computing dataset...")


        # Read data
        train_data_files = [name for name in os.listdir('data/train') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
        train_data = [pd.read_csv(f'data/train/{name}') for name in train_data_files]


        # Format data
        train_formatter = PandasFormatterEnsemble(train_data)
        train_sequences = train_formatter.format(output_format="dataclass").movements
        train_sequences = {ind : coords.to_numpy(dtype=np.float64).tolist() for ind, coords in train_sequences.items()}

        # mean_std_coords = tuple([(np.array(val).mean(), np.array(val).std()) for val in zip(*[sample for seq in train_sequences.values() for sample in seq])]) # mean and std values for each dimension
        # mean_coord_t = torch.tensor([mean_std_coords[0][0], mean_std_coords[1][0], mean_std_coords[2][0]])
        # std_coord_t = torch.tensor([mean_std_coords[0][1], mean_std_coords[1][1], mean_std_coords[2][1]])
        min_coord_x, max_coord_x, min_coord_y, max_coord_y, min_coord_z, max_coord_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for seq in train_sequences.values():
                for sample in seq:
                        min_coord_x = min(min_coord_x, sample[0])
                        max_coord_x = max(max_coord_x, sample[0])
                        min_coord_y = min(min_coord_y, sample[1])
                        max_coord_y = max(max_coord_y, sample[1])
                        min_coord_z = min(min_coord_z, sample[2])
                        max_coord_z = max(max_coord_z, sample[2])
        min_coords = torch.tensor([min_coord_x, min_coord_y, min_coord_z])
        max_coords = torch.tensor([max_coord_x, max_coord_y, max_coord_z])


        # Create dataset
        solver = DynamicsSolver(mass=1, dimensions=3)

        def transform(x_i, x_ip1, prev_v0 = None):
                x = torch.tensor(x_i).float()
                y = torch.stack([torch.tensor(x_i).float(), torch.tensor(x_ip1).float()]) # concatenate x_i and x_i+1

                prev_v0 = prev_v0.view(1,1,-1) if prev_v0 is not None else None

                # x = (x - mean_coord_t) / std_coord_t # normalize data
                # y = (y - mean_coord_t) / std_coord_t # normalize data
                x = (x - min_coords) / (max_coords - min_coords) # normalize data
                y = (y - min_coords) / (max_coords - min_coords) # normalize data

                
                force, a, v = solver.compute_force(y.unsqueeze(0), v0=prev_v0, return_velocity=True) # target data is force applied on target step (t+1), corresponds to acceleration when setting mass=1
                
                y = force[:,0,:].squeeze(0) # force applied on step x_i to reach x_i+1
                v0 = v[:,-1,:].squeeze(0) # velocity reached at step x_i+1

                return x, y, v0

        transformed_sequences = {}
        for ind, seq in train_sequences.items():
                v0 = None # Assume initial speed is 0
                for i in range(len(seq) - 1):
                        x, y, v0 = transform(seq[i], seq[i+1], v0)

                        if ind not in transformed_sequences:
                                transformed_sequences[ind] = []
                        transformed_sequences[ind].append({
                                "x": x.tolist(),
                                "a": y.tolist(),
                                "v": v0.tolist()
                        })

        train_dataset = DynamicSeriesDataset(transformed_sequences, lookback=TAU_MAX+1, target_offset_start=0, target_offset_end=0) # no offset as we want to predict the force applied on the current step

        # Save dataset
        os.makedirs("data/gen", exist_ok=True)
        torch.save(train_dataset,"data/gen/data_dynamics.pt")


# Build data loader
dataset = torch.utils.data.random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset))], generator=torch.Generator().manual_seed(1))
train_loader = DataLoader(dataset[0], batch_size=64, shuffle=True)
val_loader = DataLoader(dataset[1], batch_size=64, shuffle=False)



# # Build model
if args.save is None:
        model = MODELS[args.model_type](lookback=TAU_MAX+1)
else:
        print(f"Save provided. Loading {args.model_type} model from {args.save}...")
        model = MODELS[args.model_type].load_from_checkpoint(args.save, lookback=TAU_MAX+1)


# Train model
trainer = pl.Trainer(
        max_epochs=20,
        devices=[0],
        accelerator="gpu",
        logger=WandbLogger(name=f"{args.model_type}_train", project=args.wandb_project) if args.wandb_project else None,
)

trainer.fit(model, train_loader, val_loader)