
import torch
from torch.utils.data import Dataset


class SeriesDataset(Dataset): # Switch to pytorch_forecasting.data.timeseries.TimeSeriesDataSet if tasks become more complex (https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html)
    
    def __init__(self, sequences, tau_max):
        self.x, self.y, self.individual = self._create_dataset(sequences, tau_max)
        self.tau_max = tau_max    

    def _create_dataset(self, sequences, tau_max):
        x, y, individual = [], [], []

        for key, sequence in sequences.items():
            for i in range(len(sequence)-tau_max):
                feature = sequence[i:i+tau_max]
                target = sequence[i+1:i+tau_max+1]
                x.append(feature)
                y.append(target)
                individual.append(key)

        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), individual # (num_samples, tau_max, num_variables) (num_samples, tau_max, num_variables) (num_samples,)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.individual[idx]