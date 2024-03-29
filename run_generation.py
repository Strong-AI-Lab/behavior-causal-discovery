
import argparse
import pandas as pd
import numpy as np
import re
import os
import  tqdm
import time

from src.data.format_data import PandasFormatterEnsemble
from src.data.constants import MASKED_VARIABLES
from src.data.dataset import SeriesDataset
from src.model.behaviour_model import TSLinearCausal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.nn.functional import kl_div

from scipy import stats


# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('save', type=str, help='Load the causal graph from a save folder.')
args = parser.parse_args()

save = args.save
print(f"Arguments: save={save}")


# Read data
data_files = [name for name in os.listdir('data/test') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
data = [pd.read_csv(f'data/test/{name}') for name in data_files]
print(data)


# Set constants
TAU_MAX = 5
nb_seeds = 10
ks_threshold = 0.05


# Format data
formatter = PandasFormatterEnsemble(data)
sequences, *_ = formatter.format(event_driven=True)
sequences = {i: sequence for i, sequence in enumerate(sequences)}
variables = formatter.get_formatted_columns()
num_var = len(variables)
print(f"Graph with {num_var} variables: {variables}.")


# Create dataset
dataset = SeriesDataset(sequences, lookback=TAU_MAX+1)

assert nb_seeds <= len(dataset), f"nb_seeds ({nb_seeds}) is larger than the number of sequences ({len(dataset)})."


# Get model
print(f"Save provided. Loading results from {save}...")
val_matrix = np.load(f'{save}/val_matrix.npy')
val_matrix = torch.nan_to_num(torch.from_numpy(val_matrix).float())

graph = np.load(f'{save}/graph.npy')
graph[np.where(graph != "-->")] = "0"
graph[np.where(graph == "-->")] = "1"
graph = graph.astype(np.int64)
graph = torch.from_numpy(graph).float()

assert TAU_MAX == val_matrix.shape[2] - 1, f"tau_max ({TAU_MAX}) does not match val_matrix.shape[2] ({val_matrix.shape[2]})."

model = TSLinearCausal(num_var, TAU_MAX+1, weights=graph*val_matrix)
loader = DataLoader(dataset,sampler=RandomSampler(range(len(dataset)), num_samples=nb_seeds), batch_size=1, shuffle=False)


# Create mask
masked_idxs = [variables.index(var) for var in MASKED_VARIABLES]
masked_num_var = num_var - len(masked_idxs)
print(f"Masking {len(masked_idxs)} variables: {MASKED_VARIABLES}")


# Generate data
initial = []
generated = []
for x, y, _ in tqdm.tqdm(loader):
    y_pred = model(x)

    # Remove masked variables
    x = x[:,:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
    y_pred = y_pred[:,:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
    
    y_pred = F.gumbel_softmax(y_pred.log(), hard=True)
    y_pred = torch.cat((x[:,1:,:], y_pred[:,-1:,:]), dim=0)

    initial.append(x[0].detach().clone()) # batchsize is 1
    generated.append(y_pred[0].detach().clone())

initial = torch.stack(initial).numpy() # (nb_seeds, tau_max+1, masked_num_var)
generated = torch.stack(generated).numpy()


# Save data
# np.save(f'{save}/generated_{generation_length}_{nb_seeds}_{time.strftime("%Y%m%d-%H%M%S")}.npy', generated)
# print(f"Saved generated data to {save}/generated_{generation_length}_{nb_seeds}_{time.strftime('%Y%m%d-%H%M%S')}.npy")


# Compare distribution of generated data against original data

## Kullback-Leibler divergence
initial_samplet = torch.from_numpy(initial).float()
generated_samplet = torch.from_numpy(generated).float()

print(f"initial_samplet.shape: {initial_samplet.shape}, generated_samplet.shape: {generated_samplet.shape}")

initial_samplet = initial_samplet.mean(dim=1).view(nb_seeds, masked_num_var).clamp(min=1e-8, max=1-1e-8).log()
generated_samplet = generated_samplet.mean(dim=1).view(nb_seeds, masked_num_var).clamp(min=1e-8, max=1-1e-8).log()

div = kl_div(initial_samplet, generated_samplet, reduction='batchmean', log_target=True)
print(f"Kullback-Leibler divergence: {div}")




## Kolmogorov-Smirnov test (following <https://arize.com/blog-course/kolmogorov-smirnov-test/> and <http://daithiocrualaoich.github.io/kolmogorov_smirnov/>)

def compute_ecdf(data):
    n = np.count_nonzero(data)
    ecdf = np.cumsum(data) if n == 0 else np.cumsum(data) / n
    return ecdf
    
variable_names = [name for name in variables if name not in MASKED_VARIABLES]
for i in range(masked_num_var):
    ks_values = []
    for batch in range(nb_seeds):
        # ecdf_i = compute_ecdf(initial[batch,:,i])
        # ecdf_g = compute_ecdf(generated[batch,:,i])
        # ks = np.max(np.abs(ecdf_i - ecdf_g))

        ks = stats.kstest(generated[batch,:,i], initial[batch,:,i])

        ks_values.append(ks)

    count_above_threshold = np.count_nonzero([ks.pvalue > ks_threshold for ks in ks_values])
    print(f"Variable {variable_names[i]}: {count_above_threshold} / {nb_seeds} ({count_above_threshold / nb_seeds * 100}%) sequences are similar to the original data (pvalue above threshold {ks_threshold}).")
        










