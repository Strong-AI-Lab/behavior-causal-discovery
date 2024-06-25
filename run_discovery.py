
import argparse
import os
import time
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data.format_data import PandasFormatterEnsemble, ResultsFormatter

from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import plotting as tp
from tigramite import data_processing as pp



# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('--save', type=str, default=None, help='If provided, loads the results from a save folder instead of running the algorithm again.')
parser.add_argument('--filter', type=str, default=None, help='If provided, filters the results to only include the most significant links. Options: ' + 
                                                                '"low"  : remove links with low values (default threshold can be modified by setting `low=new_threshold`); ' +
                                                                '"high" : remove links with high values (default threshold can be modified by setting `high=new_threshold`); ' +
                                                                '"neighbor_effect" : remove links to neighbors; ' + 
                                                                '"corr" : remove correlations without causation; ' +
                                                                '"zone" : remove links to zone variables; ' +
                                                                '"zero" : remove variables with no links. Careful, the order of the filters matters. ' +
                                                                'Multiple filters can be applied by separating them with a comma.')
parser.add_argument('--skip', type=str, default=None, help='If provided, skips the specified plots. Options: ' +
                                                                '"time_series" : skip time series plot; ' +
                                                                '"correlations" : skip correlations plot; ' +
                                                                '"result_graph" : skip result graph plot; ' +
                                                                '"result_time_series_graph" : skip result time series graph plot. ' +
                                                                'Multiple plots can be skipped by separating them with a comma.')

args = parser.parse_args()
save = args.save
filter = args.filter
skips = args.skip.split(",") if args.skip is not None else []
print(f"Arguments: save={save}, filter={filter}")


# Read data
data_files = [name for name in os.listdir('data/train') if re.match(r'\d{2}-\d{2}-\d{2}_C\d_\d+.csv', name)]
data = [pd.read_csv(f'data/train/{name}') for name in data_files]
print(data)


# Set constants
TAU_MAX = 5
alpha_level = 0.05
pc_alpha = 0.05
low_filter_default = 0.075
high_filter_default = 0.925


# Format data
formatter = PandasFormatterEnsemble(data)
sequences, *_ = formatter.format(event_driven=True)
sequences = {i: sequence for i, sequence in enumerate(sequences)}
variables = formatter.get_formatted_columns()
num_variables = len(variables)
print(f"Graph with {num_variables} variables: {variables}.")


# Set variables
if save is not None:
    save = save[:-1] if save.endswith('/') else save
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
if "time_series" in skips:
    print("Skipping time series plot...")
else:
    print("Plotting time series...")
    for member in data.values.keys():
        _, axes = tp.plot_timeseries(selected_dataset = member, dataframe = data, label_fontsize = 6, figsize = (8, 80))
        for a in axes:
            a.yaxis.label.set(rotation=90)
        plt.savefig(f'{save_folder}/time_series_{member}.png')
        break


# Plot correlations
if "correlations" in skips:
    print("Skipping correlations plot...")
else:
    print("Plotting correlations...")
    correlations = pcmci.get_lagged_dependencies(tau_max=TAU_MAX, alpha_level=alpha_level, val_only=True)['val_matrix']
    matrix_lags = np.argmax(np.abs(correlations), axis=2)
    dataset_sizes = [value.shape[0] for value in data.values.values()]
    max_size_idx = dataset_sizes.index(max(dataset_sizes))
    tp.plot_scatterplots(dataframe=data, selected_dataset=max_size_idx, add_scatterplot_args={'matrix_lags':matrix_lags}, setup_args={"label_fontsize" : 6, "figsize" : (80, 80)})
    plt.savefig(f'{save_folder}/correlations.png')


# Run causal discovery algorithm
if results is None:
    print("Running causal discovery algorithm...")
    results = pcmci.run_pcmci(tau_max=TAU_MAX, alpha_level=alpha_level, pc_alpha=pc_alpha)
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
if filter is not None and not ("result_graph" in skips and "result_time_series_graph" in skips):
    for f in filter.split(","):
        print(f"Filtering results using {f}...")
        if f == 'low' or re.match(r'low=\d+(\.\d+)?', f):
            low_filter = float(re.match(r'low=(\d+(\.\d+)?)', f).group(1)) if re.match(r'low=\d+(\.\d+)?', f) else low_filter_default
            results = ResultsFormatter(results['graph'], results['val_matrix']) \
                        .low_filter(low_filter) \
                        .get_results()
        elif f == "high" or re.match(r'high=\d+(\.\d+)?', f):
            high_filter = float(re.match(r'high=(\d+(\.\d+)?)', f).group(1)) if re.match(r'high=\d+(\.\d+)?', f) else high_filter_default
            results = ResultsFormatter(results['graph'], results['val_matrix']) \
                        .high_filter(high_filter) \
                        .get_results()
        elif f == "neighbor_effect" or re.match(r'neighbor_effect=\w+', f):
            remove_bidirectional = re.match(r'neighbor_effect=(\w+)', f).group(1) == "bidirectional" if re.match(r'neighbor_effect=\w+', f) else False
            results = ResultsFormatter(results['graph'], results['val_matrix']) \
                        .var_filter([], [variables.index(v) for v in variables if v.startswith('close_neighbour_') or v.startswith('distant_neighbour_')], remove_bidirectional=remove_bidirectional) \
                        .get_results()
        elif f == "corr":
            results = ResultsFormatter(results['graph'], results['val_matrix']) \
                        .corr_filter() \
                        .get_results()
        elif f == "zone" or re.match(r'zone=\w+', f):
            remove_bidirectional = re.match(r'zone=(\w+)', f).group(1) == "bidirectional" if re.match(r'zone=\w+', f) else False
            results = ResultsFormatter(results['graph'], results['val_matrix']) \
                        .var_filter([], [variables.index(v) for v in variables if v.endswith('_zone')], remove_bidirectional=remove_bidirectional) \
                        .get_results()
        elif f == "zero":
            results = ResultsFormatter(results['graph'], results['val_matrix']) \
                        .row_filter(variables) \
                        .get_results()
            variables = results['var_names']
        elif f == "reverse":
            raise NotImplementedError("Reverse filter not implemented yet.")
        else:
            print(f"Filter {f} not recognised. Skipping filter...")


# Visualise results
if "result_graph" in skips:
    print("Skipping result graph plot...")
else:
    print("Visualising graph...")
    tp.plot_graph(graph=results['graph'], val_matrix=results['val_matrix'], var_names=variables, node_label_size = 38, label_fontsize = 45, figsize = (40, 40), node_size = 0.25, arrow_linewidth=36.0)
    plt.savefig(f'{save_folder}/result_graph.png')

if "result_time_series_graph" in skips:
    print("Skipping result time series graph plot...")
else:
    print("Visualising time series graph...")
    tp.plot_time_series_graph(val_matrix=results['val_matrix'], graph=results['graph'], var_names=variables, link_colorbar_label='MCI', label_fontsize = 40, figsize = (90, 40), node_size = 0.05, arrow_linewidth=12.0)
    plt.savefig(f'{save_folder}/result_time_series_graph.png')