
import torch


# Causal time series prediction model
class TSLinearCausal(torch.nn.Module):
    def __init__(self, num_variables, lookback, weights : torch.Tensor = None):
        super().__init__()
        self.num_variables = num_variables
        self.lookback = lookback
        self.conv = torch.nn.Conv1d(1, num_variables, num_variables*lookback, stride=num_variables, padding=num_variables*(lookback-1), bias=False)

        if weights is not None: # weight shape is (num_variables, num_variables, lookback)
            weights = weights.transpose(1,2).reshape((num_variables, 1, num_variables*lookback)).to(torch.float32)
            self.conv.weight.data = weights

    def forward(self, x):
        batch_size = x.shape[0] # x shape is (batch_size, lookback, num_variables)
        x = x.view((batch_size,1, self.num_variables*self.lookback))
        x = self.conv(x)[:,:,:self.lookback].transpose(1,2)
        # x = x.relu().softmax(dim=-1)
        x = x.relu()

        return x