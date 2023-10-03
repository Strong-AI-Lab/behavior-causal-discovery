
import torch
from torch.utils.data import Dataset


class SeriesDataset(Dataset): # Switch to pytorch_forecasting.data.timeseries.TimeSeriesDataSet if tasks become more complex (https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html)
    
    def __init__(self, sequences, lookback):
        self.x, self.y, self.individual = self._create_dataset(sequences, lookback)
        self.lookback = lookback    

    def _create_dataset(self, sequences, lookback):
        x, y, individual = [], [], []

        for key, sequence in sequences.items():
            for i in range(len(sequence)-lookback):
                feature = sequence[i:i+lookback]
                target = sequence[i+1:i+lookback+1]
                x.append(feature)
                y.append(target)
                individual.append(key)

        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), individual # (num_samples, lookback, num_variables) (num_samples, lookback, num_variables) (num_samples,)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.individual[idx]
    


class DiscriminatorDataset(Dataset):

    def __init__(self, series):
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
