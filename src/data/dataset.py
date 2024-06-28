
from typing import Callable

from dynamics.solver import DynamicsSolver
from data.structure.loaders import Loader
from data.structure.chronology import Chronology

import torch
from torch.utils.data import Dataset


class SeriesDataset(Dataset): # Switch to pytorch_forecasting.data.timeseries.TimeSeriesDataSet if tasks become more complex (https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html)
    
    def __init__(self, chronology : Chronology, struct_loader : Loader, transform : Callable = None, **kwargs):
        self.data = struct_loader.load(chronology)
        self.transform = transform

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        sample = tuple(x[idx] for x in self.data)

        if self.transform:
            sample = self.transform(sample)

        return sample
    


class DiscriminatorDataset(Dataset): # TODO: update with chronology and remove the need for generate_series functions (if possible...)

    def __init__(self, series : dict):
        self.x = []
        self.y = []
        for s in series.values():
            for y_pred, y_truth in s:
                self.x.append(y_pred)
                self.x.append(y_truth)
                self.y.append(1)
                self.y.append(0)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
