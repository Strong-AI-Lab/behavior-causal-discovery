
import argparse
import os
import time
import re
import numpy as np
import matplotlib.pyplot as plt

from data.structure.chronology import Chronology
from data.structure.loaders import BehaviourSimpleLoader
from data.constants import ALPHA_LEVEL, PC_ALPHA, LOW_FILTER_DEFAULT, HIGH_FILTER_DEFAULT, CAUSAL_GRAPH_SAVE_FOLDER_DEFAULT
from model.causal_graph_formatter import CausalGraphFormatter
from evaluate.visualisation import plot_graph_graphviz
from script_utils.data_commons import DataManager
from script_utils.parser_commons import add_loader_arguments_to_parser, add_lookback_arguments_to_parser

from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import plotting as tp
from tigramite import data_processing as pp



# Parse arguments
print("Parsing arguments..")
parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str, help='Path to the data folder.')
parser.add_argument('--model_save', type=str, default=None, help='If provided, loads the results from a save folder instead of running the algorithm again.')
parser.add_argument('--save_folder', type=str, default=CAUSAL_GRAPH_SAVE_FOLDER_DEFAULT, help='Folder to save the results.')
parser.add_argument('--filter', type=str, default=None, help='If provided, filters the results to only include the most significant links. Options: ' + 
                                                                '"low"  : remove links with low values (default threshold can be modified by setting `low=new_threshold`); ' +
                                                                '"high" : remove links with high values (default threshold can be modified by setting `high=new_threshold`); ' +
                                                                '"neighbor_effect" : remove links to neighbors; ' + 
                                                                '"corr" : remove correlations without causation; ' +
                                                                '"zone" : remove links to zone variables; ' +
                                                                '"type" : remove links to type variables; ' +
                                                                '"zero" : remove variables with no links. Careful, the order of the filters matters. ' +
                                                                'Multiple filters can be applied by separating them with a comma.')
parser.add_argument('--skip', type=str, default=None, help='If provided, skips the specified plots. Options: ' +
                                                                '"time_series" : skip time series plot; ' +
                                                                '"correlations" : skip correlations plot; ' +
                                                                '"result_graph" : skip result graph plot; ' +
                                                                '"result_time_series_graph" : skip result time series graph plot. ' +
                                                                'Multiple plots can be skipped by separating them with a comma.')
parser = add_lookback_arguments_to_parser(parser)
parser = add_loader_arguments_to_parser(parser)
args = parser.parse_args()

model_save = args.model_save
save_folder = args.save_folder
filter = args.filter
skips = args.skip.split(",") if args.skip is not None else []


# Set variables. /!\ Chronology will be re-written twice if force_data_computation is enabled.
chronology = DataManager.load_data(
    path=args.data_path,
    data_type=Chronology,
    chronology_kwargs={"fix_errors": args.fix_errors_data, "filter_null_state_trajectories": args.filter_null_state_trajectories},
    force_data_computation=args.force_data_computation,
    saving_allowed=True,
)
variables = chronology.get_labels()
num_variables = len(variables)

print(f"Graph with {num_variables} variables: {variables}.")


# Create sequences
sequences = DataManager.load_data(
    path=args.data_path,
    data_type=dict,
    loader_type=BehaviourSimpleLoader,
    chronology_kwargs={"fix_errors": args.fix_errors_data, "filter_null_state_trajectories": args.filter_null_state_trajectories},
    loader_kwargs={"skip_stationary": args.skip_stationary, "vector_columns": variables},
    force_data_computation=args.force_data_computation,
    saving_allowed=True,
)


# Load model and set save folder
if model_save is not None:
    # Load model
    model_save = os.path.normpath(model_save)
    print(f"Save provided. Loading results from {model_save}...")
    results = {
        'val_matrix': np.load(os.path.join(model_save, 'val_matrix.npy')),
        'graph': np.load(os.path.join(model_save, 'graph.npy'))
    }

    # Set save folder
    model_save_name = os.path.basename(model_save)
    model_save_name = model_save_name + '_2' if not re.match(r'.*_\d+$', model_save_name) else re.sub(r'(\d+)$', lambda x: str(int(x.group(1)) + 1), model_save_name)
    while os.path.exists(os.path.join(save_folder, model_save_name)):
        model_save_name = re.sub(r'(\d+)$', lambda x: str(int(x.group(1)) + 1), model_save_name)
    save_folder = os.path.join(save_folder, model_save_name)

else:
    # Set save folder
    print("No save provided.")
    save_folder = os.path.join(save_folder, f'save_{time.strftime("%Y%m%d-%H%M%S")}')
    results = None

print(f"Results will be saved in {save_folder}.")
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
        plt.savefig(os.path.join(save_folder, f'time_series_{member}.png'))
        break


# Plot correlations
if "correlations" in skips:
    print("Skipping correlations plot...")
else:
    print("Plotting correlations...")
    correlations = pcmci.get_lagged_dependencies(tau_max=args.tau_max, alpha_level=ALPHA_LEVEL, val_only=True)['val_matrix']
    matrix_lags = np.argmax(np.abs(correlations), axis=2)
    dataset_sizes = [value.shape[0] for value in data.values.values()]
    max_size_idx = dataset_sizes.index(max(dataset_sizes))
    tp.plot_scatterplots(dataframe=data, selected_dataset=max_size_idx, add_scatterplot_args={'matrix_lags':matrix_lags}, setup_args={"label_fontsize" : 6, "figsize" : (80, 80)})
    plt.savefig(os.path.join(save_folder, f'correlations.png'))


# Run causal discovery algorithm
if results is None:
    print("Running causal discovery algorithm...")
    results = pcmci.run_pcmci(tau_max=args.tau_max, alpha_level=ALPHA_LEVEL, pc_alpha=PC_ALPHA)
else:
    print("Skipping causal discovery algorithm...")


# Save links
print("Saving links...")
tp.write_csv(
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    var_names=variables,
    save_name=os.path.join(save_folder, 'links.csv'),
    digits=5,
)
np.save(os.path.join(save_folder, 'val_matrix.npy'), results['val_matrix'])
np.save(os.path.join(save_folder, 'graph.npy'), results['graph'])


# Filter
if filter is not None and not ("result_graph" in skips and "result_time_series_graph" in skips):
    for f in filter.split(","):
        print(f"Filtering results using {f}...")
        if f == 'low' or re.match(r'low=\d+(\.\d+)?', f):
            low_filter = float(re.match(r'low=(\d+(\.\d+)?)', f).group(1)) if re.match(r'low=\d+(\.\d+)?', f) else LOW_FILTER_DEFAULT
            results = CausalGraphFormatter(results['graph'], results['val_matrix']) \
                        .low_filter(low_filter) \
                        .get_results()
        elif f == "high" or re.match(r'high=\d+(\.\d+)?', f):
            high_filter = float(re.match(r'high=(\d+(\.\d+)?)', f).group(1)) if re.match(r'high=\d+(\.\d+)?', f) else HIGH_FILTER_DEFAULT
            results = CausalGraphFormatter(results['graph'], results['val_matrix']) \
                        .high_filter(high_filter) \
                        .get_results()
        elif f == "neighbor_effect" or re.match(r'neighbor_effect=\w+', f):
            remove_bidirectional = re.match(r'neighbor_effect=(\w+)', f).group(1) == "bidirectional" if re.match(r'neighbor_effect=\w+', f) else False
            results = CausalGraphFormatter(results['graph'], results['val_matrix']) \
                        .var_filter([], [variables.index(v) for v in variables if v.startswith('close_neighbour_') or v.startswith('distant_neighbour_')], remove_bidirectional=remove_bidirectional) \
                        .get_results()
        elif f == "corr":
            results = CausalGraphFormatter(results['graph'], results['val_matrix']) \
                        .corr_filter() \
                        .get_results()
        elif f == "zone" or re.match(r'zone=\w+', f):
            remove_bidirectional = re.match(r'zone=(\w+)', f).group(1) == "bidirectional" if re.match(r'zone=\w+', f) else False
            results = CausalGraphFormatter(results['graph'], results['val_matrix']) \
                        .var_filter([], [variables.index(v) for v in variables if v.endswith('_zone')], remove_bidirectional=remove_bidirectional) \
                        .get_results()
        elif f == "type" or re.match(r'type=\w+', f):
            remove_bidirectional = re.match(r'type=(\w+)', f).group(1) == "bidirectional" if re.match(r'type=\w+', f) else False
            results = CausalGraphFormatter(results['graph'], results['val_matrix']) \
                        .var_filter([], [variables.index(v) for v in variables if v.endswith('_type')], remove_bidirectional=remove_bidirectional) \
                        .get_results()
        elif f == "zero":
            results = CausalGraphFormatter(results['graph'], results['val_matrix']) \
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
    plt.savefig(os.path.join(save_folder, 'result_graph.png'))

    plot_graph_graphviz(results['graph'], results['val_matrix'], variables, save_folder, 'result_graph_alter_vis')

if "result_time_series_graph" in skips:
    print("Skipping result time series graph plot...")
else:
    print("Visualising time series graph...")
    tp.plot_time_series_graph(graph=results['graph'], val_matrix=results['val_matrix'], var_names=variables, link_colorbar_label='MCI', label_fontsize = 40, figsize = (90, 40), node_size = 0.05, arrow_linewidth=12.0)
    plt.savefig(os.path.join(save_folder, 'result_time_series_graph.png'))

    plot_graph_graphviz(results['graph'], results['val_matrix'], variables, save_folder, 'result_time_series_graph_alter_vis', keep_time=True)