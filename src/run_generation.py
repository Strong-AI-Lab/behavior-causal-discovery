
import argparse
import time
import os

from data.modelling.chronology_generator import ChronologyGenerator, GENERATOR_MODE_LOADER
from data.structure.chronology import Chronology
from model.behaviour_model import BEHAVIOUR_MODELS
from model.dynamics_model import DYNAMIC_MODELS
from model.graph_dynamics_model import GRAPH_DYNAMIC_MODELS
from data.constants import TAU_MAX, DATA_STRUCTURES_SAVE_FOLDER_DEFAULT


MODELS = {**DYNAMIC_MODELS, **GRAPH_DYNAMIC_MODELS, **BEHAVIOUR_MODELS}


# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('model_save', type=str, help='Load the model from a save folder.')
parser.add_argument('structure', type=str, help='Load the structure from a save folder.')
parser.add_argument('--mode', type=str, default="dynamic", help=f'Mode of the generator. Options: {",".join(GENERATOR_MODE_LOADER.keys())}.')
parser.add_argument('--model_type',type=str, default="dynamical_lstm", help=f'Type of model to use. Options: {",".join(MODELS.keys())}.')
parser.add_argument('--nb_steps', type=int, default=1, help='Number of steps to generate.')
parser.add_argument('--tau_max', type=int, default=TAU_MAX, help='Maximum lag to consider.')
parser.add_argument('--save_folder', type=str, default=DATA_STRUCTURES_SAVE_FOLDER_DEFAULT, help='Folder to save the results.')
args = parser.parse_args()


# Load structure
chronology = Chronology.deserialize(args.structure)
end_time = chronology.end_time
_, chronology = chronology.split(end_time-2*(args.tau_max+1)) # Keep only the last 2*(TAU_MAX+1) snapshots


# Load models
print("Loading model..")
model = MODELS[args.model_type].load_from_checkpoint(args.model_save)


# Create generator
print("Creating generator..")
generator = ChronologyGenerator(chronology=chronology, models=model, lookback=args.tau_max+1, skip_stationary=True, modes=args.mode)


# Run data generation
print("Generating data..")
generator.generate(args.nb_steps)
str_time = time.strftime("%Y%m%d-%H%M%S")
generator.save(os.path.join(args.save_folder, f"chronology_generated_{args.mode}_{args.model_type}_{str_time}.json"))