
import numpy as np
import torch

from model.causal_graph_formatter import CausalGraphFormatter

LOW_FILTER = 0.075


def load_graph(savefile : str, variables : list, filter : list, edge_type : str) -> torch.Tensor:
    val_matrix = np.load(f'{savefile}/val_matrix.npy')
    graph = np.load(f'{savefile}/graph.npy')

    for f in filter:
        print(f"Filtering results using {f}...")
        if f == 'low':
            filtered_values = CausalGraphFormatter(graph, val_matrix).low_filter(LOW_FILTER)
            val_matrix = filtered_values.get_val_matrix()
            graph = filtered_values.get_graph()
        elif f == "neighbor_effect":
            filtered_values = CausalGraphFormatter(graph, val_matrix).var_filter([], [variables.index(v) for v in variables if v.startswith('close_neighbour_') or v.startswith('distant_neighbour_')])
            val_matrix = filtered_values.get_val_matrix()
            graph = filtered_values.get_graph()
        elif f == "corr":
            filtered_values = CausalGraphFormatter(graph, val_matrix).corr_filter()
            val_matrix = filtered_values.get_val_matrix()
            graph = filtered_values.get_graph()
        else:
            print(f"Filter {f} not recognised. Skipping filter...")
                    
    val_matrix = torch.nan_to_num(torch.from_numpy(val_matrix).float())
    graph[np.where(graph != "-->")] = "0"
    graph[np.where(graph == "-->")] = "1"
    graph = graph.astype(np.int64)
    graph = torch.from_numpy(graph).float()

    if edge_type == "all":
        graph_weights=graph*val_matrix
    elif edge_type == "coefficients":
        graph_weights=val_matrix
    elif edge_type == "edges":
        graph_weights=graph
    else:
        raise ValueError(f"causal_graph must be one of 'all', 'coefficients', 'edges'. Got {edge_type}.")
    
    return graph_weights