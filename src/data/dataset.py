
from typing import Callable

from dynamics.solver import DynamicsSolver

import torch
from torch.utils.data import Dataset


class SeriesDataset(Dataset): # Switch to pytorch_forecasting.data.timeseries.TimeSeriesDataSet if tasks become more complex (https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html)
    
    def __init__(self, sequences : dict, lookback : int, target_offset_start : int = 1, target_offset_end : int = 1, transform : Callable = None, **kwargs):
        self.data_arrays = self._create_dataset(sequences, lookback, target_offset_start, target_offset_end,**kwargs)
        self.nb_arrays = len(self.data_arrays)
        self.lookback = lookback
        self.target_offset_start = target_offset_start
        self.target_offset_end = target_offset_end
        self.transform = transform

    def _create_dataset(self, sequences : dict, lookback : int, target_offset_start : int = 1, target_offset_end : int = 1):
        x, y, individual = [], [], []

        for key, sequence in sequences.items():
            for i in range(len(sequence)-lookback-target_offset_end+1):
                feature = sequence[i:i+lookback]
                target = sequence[i+target_offset_start:i+lookback+target_offset_end]
                x.append(feature)
                y.append(target)
                individual.append(key)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), individual # (num_samples, lookback, num_variables) (num_samples, lookback, num_variables) (num_samples,)
    
    def __len__(self):
        return len(self.data_arrays[0])
    
    def __getitem__(self, idx):
        sample = tuple(x[idx] for x in self.data_arrays)

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class DynamicSeriesDataset(SeriesDataset):

    def _compute_min_max(self, sequences : dict):
        mins = [0.0] * len(sequences[0][0])
        maxs = [0.0] * len(sequences[0][0])
        for seq in sequences.values():
                for sample in seq:
                        for i in range(len(mins)):
                            mins[i] = min(mins[i], sample[i])
                            maxs[i] = max(maxs[i], sample[i])
        return torch.tensor(mins), torch.tensor(maxs)

    def _dyn_transform(x_i, x_ip1, prev_v0 = None, min_coords = None, max_coords = None, solver=None):
                x = torch.tensor(x_i).float()
                y = torch.stack([torch.tensor(x_i).float(), torch.tensor(x_ip1).float()]) # concatenate x_i and x_i+1

                prev_v0 = prev_v0.view(1,1,-1) if prev_v0 is not None else None

                if min_coords is not None and max_coords is not None:
                    # x = (x - mean_coord_t) / std_coord_t # normalize data
                    # y = (y - mean_coord_t) / std_coord_t # normalize data
                    x = (x - min_coords) / (max_coords - min_coords) # normalize data
                    y = (y - min_coords) / (max_coords - min_coords) # normalize data

                if solver is not None:
                    force, a, v = solver.compute_force(y.unsqueeze(0), v0=prev_v0, return_velocity=True) # target data is force applied on target step (t+1), corresponds to acceleration when setting mass=1
                    y = force[:,0,:].squeeze(0) # force applied on step x_i to reach x_i+1
                    v0 = v[:,-1,:].squeeze(0) # velocity reached at step x_i+1
                else:
                    y = y[:,1,:].squeeze(0)
                    v0 = y - x

                return x, y, v0

    def _create_dataset(self, sequences : dict, lookback : int, target_offset_start : int = 1, target_offset_end : int = 1):
        
        dim = sequences[0][0].shape[0]
        min_coords, max_coords = self._compute_min_max(sequences)

        solver=DynamicsSolver(mass=1, dimensions=dim)

        transformed_sequences = {}
        for ind, seq in sequences.items():
                v0 = None # Assume initial speed is 0
                for i in range(len(seq) - 1):
                        x, y, v0 = DynamicSeriesDataset._dyn_transform(seq[i], seq[i+1], v0, min_coords, max_coords, solver)

                        if ind not in transformed_sequences:
                                transformed_sequences[ind] = []
                        transformed_sequences[ind].append({
                                "x": x.tolist(),
                                "a": y.tolist(),
                                "v": v0.tolist()
                        })

        x, v, y, individual = [], [], [], []
        for key, sequence in transformed_sequences.items():
            for i in range(len(sequence)-lookback-target_offset_end+1):
                feature = sequence[i:i+lookback]
                target = sequence[i+target_offset_start:i+lookback+target_offset_end]

                coordinates = [f["x"] for f in feature]
                velocity = [f["v"] for f in feature]
                forces = [t["a"] for t in target]

                x.append(coordinates)
                v.append(velocity)
                y.append(forces)
                individual.append(key)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(v, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), individual # coordinates: (num_samples, lookback, num_variables), velocity: (num_samples, lookback, num_variables), target force: (num_samples, lookback, num_variables), individuals: (num_samples,)
    

class DynamicGraphSeriesDataset(DynamicSeriesDataset):

    def _complete_snippet_with_all_inds(self, snippet : list, individuals_list : list, feature : str, dimensions : int):
        snippet_size = len(snippet)
        res = [[[0.0]*dimensions for _ in range(len(individuals_list))] for _ in range(snippet_size)]
        for i in range(snippet_size):
            for ind in snippet[i]:
                idx = individuals_list.index(ind)
                res[i][idx] = snippet[i][ind][feature]
        return res
    
    def _to_index_vector(self, indices : list, vector_size : int):
        res = [0.0] * vector_size
        for idx in indices:
            res[indices.index(idx)] = 1.0
        return res
         

    def _create_dataset(self, sequences : dict, lookback : int, target_offset_start : int = 1, target_offset_end : int = 1, adjacency_list : dict = None):
        if adjacency_list is None:
            raise ValueError("Adjacency list is required for graph models.")
        
        dim = len(sequences[0][0])
        min_coords, max_coords = self._compute_min_max(sequences)

        solver=DynamicsSolver(mass=1, dimensions=dim)

        time_ind_series = {} # {time: {individual: [x, v, a]}}
        for ind, seq in sequences.items():
                v0 = None # Assume initial speed is 0
                for i in range(len(seq) - 1):
                        x, y, v0 = DynamicSeriesDataset._dyn_transform(seq[i], seq[i+1], v0, min_coords, max_coords, solver)

                        idx = adjacency_list[ind][i][0] # Get time index
                        if idx not in time_ind_series:
                                time_ind_series[idx] = {}
                        if ind in time_ind_series[idx]:
                                raise ValueError(f"Individual {ind} already exists at time index {idx}.")

                        time_ind_series[idx][ind] = {
                                "x": x.tolist(),
                                "a": y.tolist(),
                                "v": v0.tolist()
                        }

        if list(time_ind_series.keys()) != list(range(min(time_ind_series.keys()), max(time_ind_series.keys())+1)):
            discontinuities = sorted(set(range(min(time_ind_series.keys()), max(time_ind_series.keys())+1)) - set(time_ind_series.keys()))
            print(f"Warning! Time indices are not continuous. Missing {len(discontinuities)} indices: {discontinuities}. Expected all indices in range({min(time_ind_series.keys())}, {max(time_ind_series.keys())+1}).")

        # Convert dict to list
        time_ind_sequences = [time_ind_series[i] for i in time_ind_series.keys()]

        # Create dataset
        nb_individuals = len(sequences)
        x, v, g, y, individuals = [], [], [], [], []
        for i in range(len(time_ind_sequences)-lookback-target_offset_end+1):
            feature = time_ind_sequences[i:i+lookback]
            target = time_ind_sequences[i+target_offset_start:i+lookback+target_offset_end]

            coordinates = self._complete_snippet_with_all_inds(feature, list(sequences.keys()), "x", dim)
            velocity = self._complete_snippet_with_all_inds(feature, list(sequences.keys()), "v", dim)
            forces = self._complete_snippet_with_all_inds(target, list(sequences.keys()), "a", dim)

            x.append(coordinates)
            v.append(velocity)
            y.append(forces)
            individuals.append([self._to_index_vector(list(feature[j].keys()), nb_individuals) for j in range(lookback)])

            g_lookback = [[[0.0]*nb_individuals]*nb_individuals]*lookback
            for j in range(lookback):
                inds = list(feature[j].keys())
                for ind1 in inds:
                    try:
                        ind1_idx = list(sequences.keys()).index(ind1)
                        adj_idx = [a[0] for a in adjacency_list[ind1]][i+j]

                        close_neighbours = adjacency_list[ind1_idx][adj_idx][1]
                        dist_neighbours = adjacency_list[ind1_idx][adj_idx][2]
                        for ind2 in close_neighbours:
                            ind2_idx = list(sequences.keys()).index(ind2)
                            g_lookback[j][ind1_idx][ind2_idx] = 1.0
                        for ind2 in dist_neighbours:
                            ind2_idx = list(sequences.keys()).index(ind2)
                            g_lookback[j][ind1_idx][ind2_idx] = 0.5
                    except IndexError as e:
                        print(f"Index Error: {e}. Ignoring as due to missing data.")
            g.append(g_lookback)
        
        # coordinates: (num_samples, lookback, num_individuals, num_variables)
        # velocity: (num_samples, lookback, num_individuals, num_variables)
        # adjacency: (num_samples, lookback, num_individuals, num_individuals)
        # target force: (num_samples, lookback, num_individuals, num_variables)
        # individuals: (num_samples, lookback, nb_individuals)
        return torch.tensor(x, dtype=torch.float32), \
                torch.tensor(v, dtype=torch.float32), \
                torch.tensor(g, dtype=torch.float32), \
                torch.tensor(y, dtype=torch.float32), \
                torch.tensor(individuals, dtype=torch.int64) 
    


class DiscriminatorDataset(Dataset):

    def __init__(self, series : dict):
        self.x = []
        self.y = []
        for _, s in series.items():
            for y_pred, y_truth in s:
                self.x.append(y_pred)
                self.x.append(y_truth)
                self.y.append(1)
                self.y.append(0)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
