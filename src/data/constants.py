
import os

# Constants for model settings
TAU_MAX = 5


# Constants for causal graph
ALPHA_LEVEL = 0.05
PC_ALPHA = 0.05
LOW_FILTER_DEFAULT = 0.075
HIGH_FILTER_DEFAULT = 0.925


# Constants for visualisation
MAX_CLUSTER_DATA_POINTS = 2000


# Saving constants
RESULTS_SAVE_FOLDER_DEFAULT = "results"
CAUSAL_GRAPH_SAVE_FOLDER_DEFAULT = "causal_graphs"
DATA_STRUCTURES_SAVE_FOLDER_DEFAULT = os.path.join("data","gen")
VISUALISATION_SAVE_FOLDER_DEFAULT = "visualisations"

