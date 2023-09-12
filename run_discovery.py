
import argparse
import os
import time
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from format_data import PandasFormatterEnsemble, ResultsFormatter

from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import plotting as tp
from tigramite import data_processing as pp



# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('--save', type=str, default=None, help='If provided, loads the results from a save folder instead of running the algorithm again.')
parser.add_argument('--filter', type=str, default=None, help='If provided, filters the results to only include the most significant links. Options: ' + 
                                                                '"low"  : remove links with low values; ' +
                                                                '"neighbor_effect" : remove links to neighbors, ' + 
                                                                '"corr" : remove correlations without causation. ' +
                                                                'Multiple filters can be applied by separating them with a comma.')
args = parser.parse_args()
save = args.save
filter = args.filter
print(f"Arguments: save={save}, filter={filter}")


# Read data
data_files = [name for name in os.listdir('data/train') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
data = [pd.read_csv(f'data/train/{name}') for name in data_files]
print(data)


# Set constants
tau_max = 5
alpha_level = 0.05
pc_alpha = 0.05
low_filter = 0.075


# Format data
formatter = PandasFormatterEnsemble(data)
sequences = formatter.format(event_driven=True)
sequences = {i: sequence for i, sequence in enumerate(sequences)}
variables = formatter.get_formatted_columns()
num_var = len(variables)
print(f"Graph with {num_var} variables: {variables}.")


# Set variables
if save is not None:
    print(f"Save provided. Loading results from {save}...")
    results = {
        'val_matrix': np.load(f'{save}/val_matrix.npy'),
        'graph': np.load(f'{save}/graph.npy')
    }
    save_folder = save + '_2' if not re.match(r'.*_\d+$', save) else re.sub(r'(\d+)$', lambda x: str(int(x.group(1)) + 1), save)
    while os.path.exists(save_folder):
        save_folder = re.sub(r'(\d+)$', lambda x: str(int(x.group(1)) + 1), save_folder)
else:
    print("No save provided. Setting variables...")
    save_folder = f'saves/save_{time.strftime("%Y%m%d-%H%M%S")}'
    results = None

os.makedirs(save_folder)
data = pp.DataFrame(sequences, analysis_mode = 'multiple', var_names=variables)
pcmci = PCMCI(dataframe=data, cond_ind_test=ParCorr())


# Plot time series
print("Plotting time series...")
for member in data.values.keys():
    _, axes = tp.plot_timeseries(selected_dataset = member, dataframe = data, label_fontsize = 6, figsize = (8, 80))
    for a in axes:
        a.yaxis.label.set(rotation=90)
    plt.savefig(f'{save_folder}/time_series_{member}.png')
    break


# Plot correlations
print("Plotting correlations...")
correlations = pcmci.get_lagged_dependencies(tau_max=tau_max, alpha_level=alpha_level, val_only=True)['val_matrix']
matrix_lags = np.argmax(np.abs(correlations), axis=2)
dataset_sizes = [value.shape[0] for value in data.values.values()]
max_size_idx = dataset_sizes.index(max(dataset_sizes))
tp.plot_scatterplots(dataframe=data, selected_dataset=max_size_idx, add_scatterplot_args={'matrix_lags':matrix_lags}, setup_args={"label_fontsize" : 6, "figsize" : (80, 80)})
plt.savefig(f'{save_folder}/correlations.png')


# Run causal discovery algorithm
if results is None:
    print("Running causal discovery algorithm...")
    results = pcmci.run_pcmci(tau_max=tau_max, alpha_level=alpha_level, pc_alpha=pc_alpha)
else:
    print("Skipping causal discovery algorithm...")


# Save links
print("Saving links...")
tp.write_csv(
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    var_names=variables,
    save_name=f'{save_folder}/links.csv',
    digits=5,
)
np.save(f'{save_folder}/val_matrix.npy', results['val_matrix'])
np.save(f'{save_folder}/graph.npy', results['graph'])


# Filter
if filter is not None:
    for f in filter.split(","):
        print(f"Filtering results using {f}...")
        if f == 'low':
            results = ResultsFormatter(results['graph'], results['val_matrix']) \
                        .low_filter(low_filter) \
                        .get_results()
        elif f == "neighbor_effect":
            results = ResultsFormatter(results['graph'], results['val_matrix']) \
                        .var_filter([], [variables.index(v) for v in variables if v.startswith('close_neighbour_') or v.startswith('distant_neighbour_')]) \
                        .get_results()
        elif f == "corr":
            results = ResultsFormatter(results['graph'], results['val_matrix']) \
                        .corr_filter() \
                        .get_results()
        else:
            print(f"Filter {f} not recognised. Skipping filter...")


# Visualise results
print("Visualising results...")
tp.plot_graph(graph=results['graph'], val_matrix=results['val_matrix'], var_names=variables, label_fontsize = 15, figsize = (40, 40), node_size = 0.05)
plt.savefig(f'{save_folder}/result_graph.png')

tp.plot_time_series_graph(val_matrix=results['val_matrix'], graph=results['graph'], var_names=variables, link_colorbar_label='MCI', label_fontsize = 15, figsize = (40, 40), node_size = 0.01)
plt.savefig(f'{save_folder}/result_time_series_graph.png')