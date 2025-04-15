
import argparse

from data.constants import TAU_MAX


def add_causal_arguments_to_parser(parser : argparse.ArgumentParser):
    parser.add_argument('--filter', type=str, default=None, help='If provided, filters the causal graph to only include the most significant links. Options: ' + 
                                                                '"low" : remove links with low values; ' +
                                                                '"neighbor_effect" : remove links to neighbors, ' + 
                                                                '"corr" : remove correlations without causation. ' +
                                                                'Multiple filters can be applied by separating them with a comma.')
    return parser


def add_loader_arguments_to_parser(parser : argparse.ArgumentParser):
    parser.add_argument('--force_data_computation', action="store_true", help='If specified, forces the computation of the force data from the raw data.')
    parser.add_argument('--fix_errors_data', action="store_true", help='If specified, fixes simple errors and fills missing values in the data using estimation heuristics.')
    parser.add_argument('--filter_null_state_trajectories', action="store_true", help='If specified, removes trajectories with null states from data.')
    parser.add_argument('--do_not_skip_stationary', action="store_false", dest="skip_stationary", help='If specified, does not skip stationary trajectories when loading data.')
    return parser


def add_lookback_arguments_to_parser(parser : argparse.ArgumentParser):
    parser.add_argument('--tau_max', type=int, default=TAU_MAX, help='Maximum lag to consider.')
    return parser

