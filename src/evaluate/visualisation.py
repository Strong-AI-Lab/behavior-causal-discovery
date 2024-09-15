
import os
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import numpy as np
import networkx as nx

from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.manifold import MDS
import torch


COLOR_TAGS = [
    "rgba(31, 119, 180, 0.8)",
    "rgba(255, 127, 14, 0.8)",
    "rgba(44, 160, 44, 0.8)",
    "rgba(214, 39, 40, 0.8)",
    "rgba(148, 103, 189, 0.8)",
    "rgba(140, 86, 75, 0.8)",
    "rgba(227, 119, 194, 0.8)",
    "rgba(127, 127, 127, 0.8)",
    "rgba(188, 189, 34, 0.8)",
    "rgba(23, 190, 207, 0.8)",
    "rgba(31, 119, 180, 0.8)",
    "rgba(255, 127, 14, 0.8)",
    "rgba(44, 160, 44, 0.8)",
    "rgba(214, 39, 40, 0.8)",
    "rgba(148, 103, 189, 0.8)",
    "rgba(140, 86, 75, 0.8)",
    "rgba(227, 119, 194, 0.8)",
    "rgba(127, 127, 127, 0.8)",
    "rgba(188, 189, 34, 0.8)",
    "rgba(23, 190, 207, 0.8)",
    "rgba(31, 119, 180, 0.8)"
    ] # taken from https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json


def flatten_time_graph(graph : np.ndarray):
    # [nb_variables, nb_variables, tau] -> [nb_variables*tau, nb_variables*tau]
    nb_variables = graph.shape[0]
    tau = graph.shape[-1]
    new_graph = np.zeros((tau*nb_variables, tau*nb_variables))
    for i in range(tau):
        for j in range(i, tau):
            new_graph[i*nb_variables:(i+1)*nb_variables, j*nb_variables:(j+1)*nb_variables] = graph[:, :, i] # add links from timestep i to timestep j
    return new_graph


def prep_labels(labels : List[str], max_length : int = 12):
    new_labels = []
    label_lengths = []
    for label in labels:
        words = label.split("_")
        words[0] = words[0].capitalize()
        lines = []
        line = ""
        for word in words:
            if len(line) > 0 and len(line) + len(word) > max_length:
                lines.append(line)
                line = ""
            line += f"{word} "
        lines.append(line)
        label_lengths.append(max([len(line) for line in lines]))
        new_labels.append("\n".join(lines))
    return new_labels, label_lengths


def plot_graph_graphviz(graph : np.ndarray, val_matrix : np.ndarray, var_names : List[str], save_folder : str, save_file : str, keep_time : bool = False):
    # Create adjacency matrix
    graph = np.copy(graph)
    graph[np.where(graph == "-->")] = "1"
    graph[np.where(graph == "o-o")] = "1"
    graph[np.where(graph == "")] = "0"
    graph = graph.astype(np.int64)
    graph=graph*val_matrix
    graph=np.nan_to_num(graph)

    # Remove time dimension
    if not keep_time:
        graph = graph.sum(axis=-1) # Flatten time dimension
    else:
        tau = graph.shape[-1]
        graph = flatten_time_graph(graph) # Expand time dimension (add time sensitive variables to flattened graph)
        var_names = [f"{var}_{i}" for i in range(tau) for var in var_names]
    np.fill_diagonal(graph, 0) # Remove self loops


    # Create plot
    ng=nx.from_numpy_array(graph, create_using=nx.DiGraph)
    new_labels, label_lengths = prep_labels(var_names)
    ng=nx.relabel_nodes(ng, dict(enumerate(new_labels)))
    edge_colors = dict([((u, v,), attrs['weight']) for u, v, attrs in ng.edges(data=True)])
    node_sizes = [400*(i+1) for i in label_lengths]
    nb_nodes = len(ng.nodes)
    
    plt.figure(figsize=(10 + int(np.log(nb_nodes)), 10 + int(np.log(nb_nodes))))
    pos = nx.spring_layout(ng,scale=2)
    nx.draw(ng, pos, with_labels=True, node_size=node_sizes, node_color='white', edgecolors='black', font_size=10, font_family='serif', font_color='black', edge_color=list(edge_colors.values()), edge_cmap=cm.coolwarm, width=3, arrowsize=15)
    plt.savefig(os.path.join(save_folder, f"{save_file}.png"), bbox_inches='tight')


def generate_time_occurences(series : Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]], predicted_variable_names : List[str], save : str, nb_variables : int, min_length : int, prefix : Optional[str] = None):
    x_axis = [i for _ in range(nb_variables) for i in range(min_length)]
    y_axis = [i for i in range(nb_variables) for _ in range(min_length)]
    area_pred = [0] * (nb_variables * min_length)
    area = [0] * (nb_variables * min_length)

    for ind, s in series.items():
        for i, (y_pred, y) in enumerate(s):
            if i >= min_length:
                break
            area_pred[y_pred.argmax(dim=-1).tolist() * min_length + i] += 1
            area[y.argmax(dim=-1).tolist() * min_length + i] += 1

    max_val = max(max(area_pred), max(area))
    for save_file, results in [('prediction_time_occurences', area_pred), ('true_time_occurences', area)]:
        fig, ax = plt.subplots()
        ax.set_yticks(np.arange(nb_variables))
        ax.set_yticklabels(predicted_variable_names, rotation=35)
        ax.set_ylabel('Behaviour')
        ax.set_xlabel('Time')
        sc = ax.scatter(x_axis, y_axis, s=[r**1.5 for r in results], c=results, alpha=0.5, vmin=0, vmax=max_val)
        ax.legend(*sc.legend_elements(), loc="upper right", title="Occurence", bbox_to_anchor=(1.15, 1.05))

        if prefix is not None:
            save_file = f"{prefix}_{save_file}"
        plt.savefig(os.path.join(save, f"{save_file}.png"), bbox_inches='tight')



def generate_sankey(series : Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]], predicted_variable_names, save : str, nb_variables : int, min_length : int, prefix : Optional[str] = None):
    for save_file, truth in [('prediction_sankey', 0), ('true_sankey', 1)]:
        labels = []
        colors = []
        sources = []
        targets = []
        values = []
        flattened_values = [0] * (nb_variables * nb_variables)
        link_colors = []
        for t in range(min_length):
            nodes = list(range(nb_variables))
            labels.extend([f"{t}_{predicted_variable_names[v]}" for v in nodes])
            colors.extend([COLOR_TAGS[v] for v in nodes])
            if t < min_length - 1:
                sources.extend([t * nb_variables + v for v in nodes for _ in range(nb_variables)])
            if t > 0:
                targets.extend([t * nb_variables + v for _ in range(nb_variables) for v in nodes])
                vals = [0] * (nb_variables * nb_variables)
                for i, s in series.items():
                    if t >= len(s):
                        continue
                    x = s[t-1][truth].argmax(dim=-1).tolist()
                    y = s[t][truth].argmax(dim=-1).tolist()
                    vals[x * nb_variables + y] += 1
                values.extend(vals)
                flattened_values = [flattened_values[i] + vals[i] for i in range(nb_variables * nb_variables)]
                link_colors.extend([COLOR_TAGS[v] for v in range(nb_variables) for _ in range(nb_variables)])
            
        # Full sequence diagram
        fig = go.Figure(data=[go.Sankey(
            node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.1),
            label = [label[2:] for label in labels[:nb_variables]],
            color = colors
            ),
            link = dict(
            source = sources,
            target = targets,
            value = values,
            color = link_colors
        ))]) 
        fig.update_layout(title_text=save_file, font_size=38, width=3600, height=1800)

        if prefix is not None:
            save_file = f"{prefix}_{save_file}"
        fig.write_image(os.path.join(save, f"{save_file}_full.png"))
            
        # Flattened diagram
        fig_flat = go.Figure(data=[go.Sankey(
            node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.1),
            label = [label[2:] for label in labels[:nb_variables]],
            color = colors
            ),
            link = dict(
            source = sources[:nb_variables*nb_variables],
            target = targets[:nb_variables*nb_variables],
            value = flattened_values,
            color = link_colors[:nb_variables*nb_variables]
        ))]) 
        fig_flat.update_layout(title_text=save_file, font_size=38, width=900, height=1800)
        fig_flat.write_image(os.path.join(save, f"{save_file}_flat.png"))



CLUSTERING_ALGORITHMS = {
    "Bisecting K-Means": BisectingKMeans,
    "K-Means": KMeans,
}
def generate_clusters(series : Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]], save : str, nb_variables, cluster_lists : Optional[List[int]] = None, prefix : Optional[str] = None):
    if cluster_lists is None:
        cluster_lists = [4, 8, 16]
    
    data_pred = torch.stack([y_pred.view((nb_variables,)) for s in series.values() for y_pred, y in s]).detach().numpy()
    data_truth = torch.stack([y.view((nb_variables,)) for s in series.values() for y_pred, y in s]).detach().numpy()

    embedding = MDS(n_components=2)
    for data, file_name in [(data_pred, "prediction_clusters"), (data_truth, "true_clusters")]: # predicted series + ground truth series
        data_embed = embedding.fit_transform(data)
        fig, axs = plt.subplots(
            len(CLUSTERING_ALGORITHMS), len(cluster_lists), figsize=(12, 5)
        )
        axs = axs.T
        for i, (algorithm_name, Algorithm) in enumerate(CLUSTERING_ALGORITHMS.items()):
            for j, n_clusters in enumerate(cluster_lists):
                        algo = Algorithm(n_clusters=n_clusters, n_init=3)
                        # algo.fit(data)
                        algo.fit(data_embed)
                        centers = algo.cluster_centers_

                        axs[j, i].scatter(data_embed[:, 0], data_embed[:, 1], s=10, c=algo.labels_)
                        axs[j, i].scatter(centers[:, 0], centers[:, 1], c="r", s=20)
                        axs[j, i].set_title(f"{algorithm_name} : {n_clusters} clusters")

        if prefix is not None:
            file_name = f"{prefix}_{file_name}"
        plt.savefig(os.path.join(save, f"{file_name}.png"))

    