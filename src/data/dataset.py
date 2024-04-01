
from typing import Callable

import torch
from torch.utils.data import Dataset


class SeriesDataset(Dataset): # Switch to pytorch_forecasting.data.timeseries.TimeSeriesDataSet if tasks become more complex (https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html)
    
    def __init__(self, sequences : dict, lookback : int, target_offset_start : int = 1, target_offset_end : int = 1, transform : Callable = None):
        self.data_arrays = self._create_dataset(sequences, lookback, target_offset_start, target_offset_end)
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

    def __init__(self, sequences : dict, lookback : int, target_offset_start : int = 1, target_offset_end : int = 1, transform : Callable = None):
        super().__init__(sequences, lookback, target_offset_start, target_offset_end, transform)

    def _create_dataset(self, sequences : dict, lookback : int, target_offset_start : int = 1, target_offset_end : int = 1):
        x, v, y, individual = [], [], [], []

        for key, sequence in sequences.items():
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
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(v, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), individual # (num_samples, lookback, num_variables) (num_samples, lookback, num_variables) (num_samples,)
    


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
