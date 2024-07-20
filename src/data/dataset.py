
from typing import Callable, Optional, Tuple, Union, List, Any

from data.structure.loaders import Loader
from data.structure.chronology import Chronology

import torch
from torch.utils.data import Dataset


class SeriesDataset(Dataset): # Switch to pytorch_forecasting.data.timeseries.TimeSeriesDataSet if tasks become more complex (https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html)
    
    def __init__(self, 
                 data : Optional[Union[List[Any],Tuple[torch.Tensor],Tuple[List[torch.Tensor]]]] = None,
                 chronology : Optional[Chronology] = None, 
                 struct_loader : Optional[Loader] = None, 
                 transform : Optional[Callable] = None, 
                 **kwargs) -> None:
        if data is not None and chronology is None and struct_loader is None:
            self.data = data
        elif chronology is not None and struct_loader is not None:
            self.data = struct_loader.load(chronology, **kwargs)
        else:
            raise ValueError("Either data or chronology and struct_loader must be provided. data and chronology+struct_loader are mutually exclusive.")
        self.transform = transform

    def __len__(self) -> int:
        if isinstance(self.data, tuple):
            return len(self.data[0])
        else:
            return len(self.data)
    
    def __getitem__(self, idx) -> Union[Any, torch.Tensor, Tuple[torch.Tensor]]:
        if isinstance(self.data, tuple):
            sample = tuple(x[idx] for x in self.data)
        else:
            sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def save(self, path : str) -> None:
        torch.save(self.data, path)

    @classmethod
    def load(cls, path : str, transform : Optional[Callable] = None, **kwargs) -> 'SeriesDataset':
        data = torch.load(path)
        return cls(data, transform=transform, **kwargs)

