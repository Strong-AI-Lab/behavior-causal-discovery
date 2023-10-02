
import torch
import pytorch_lightning as pl


# Causal time series prediction model
class TSLinearCausal(torch.nn.Module):
    """
    Causal time series prediction model.
    The output corresponds to the series prediction one step in the future, e.g. if input = [x(t-2), x(t-1), x(t)], then output = [^x(t-1), ^x(t), ^x(t+1)].
    Output is generated based on the cumulated previous values in the series up to lookback. Output is unnormalised. For example:
    ```
    w=torch.nn.functional.one_hot(torch.arange(4)).unsqueeze(0).repeat(3,1,1).permute(1,2,0) # identity matrix
    model=TSLinearCausal(4,3,w) # 4 variables, lookback of 3, identity matrix as weights

    x = torch.nn.functional.one_hot(torch.arange(4)).unsqueeze(0)[:,:3,:].float()
    >>> x
    tensor([[[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.]]])
    >>> model(x)
    tensor([[[1., 0., 0., 0.],
            [1., 1., 0., 0.],
            [1., 1., 1., 0.]]])

    y = torch.zeros((1,3,4))
    y[:,:,2] = 1
    >>> y
    tensor([[[0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.]]])
    >>> model(y)
    tensor([[[0., 0., 1., 0.],
            [0., 0., 2., 0.],
            [0., 0., 3., 0.]]])
    ```
    """
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
    


# LSTM prediction model
class LSTMPredictor(pl.LightningModule):
    def __init__(self, num_var, tau_max, hidden_size=128, num_layers=6):
        super().__init__()
        self.num_var = num_var
        self.tau_max = tau_max
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(input_size=num_var, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, num_var)
        self.save_hyperparameters()

    def forward(self, x):
        batch_size = x.shape[0]
        x, _ = self.lstm(x)
        x = self.linear(x.reshape((batch_size*self.tau_max, self.hidden_size)))
        x = x.reshape((batch_size, self.tau_max, self.num_var))
        return x
    
    def training_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y)
        self.log('val_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y, i = batch
        return self(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def backward(self, loss):
        loss.backward(retain_graph=True)
    

# Transformer prediction model
class TransformerPredictor(pl.LightningModule):
    def __init__(self, num_var, tau_max, nhead=3, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.num_var = num_var
        self.tau_max = tau_max
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.transformer = torch.nn.Transformer(d_model=num_var, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=True)
        self.linear = torch.nn.Linear(tau_max*num_var, num_var)
        self.save_hyperparameters()
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.transformer(x, x)
        x = self.linear(x.view(batch_size, self.tau_max*self.num_var))
        return x
    
    def training_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y)
        self.log('val_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y, i = batch
        return self(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)



# LSTM Model that discriminates between real and generated time series
class LSTMDiscriminator(pl.LightningModule):
    def __init__(self, num_var, tau_max, hidden_size=32, num_layers=1):
        super().__init__()
        self.num_var = num_var
        self.tau_max = tau_max
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(input_size=num_var, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = torch.nn.Linear(tau_max*hidden_size, 1)
        self.save_hyperparameters()

    def forward(self, x):
        batch_size = x.shape[0]
        x, _ = self.lstm(x)
        x = self.linear(x.reshape((batch_size, self.tau_max*self.hidden_size)))
        return x
    
    def training_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y.unsqueeze(-1).float())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y.unsqueeze(-1).float())
        self.log('val_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y, i = batch
        return self(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def backward(self, loss):
        loss.backward(retain_graph=True)


# Transformer Model that discriminates between real and generated time series
class TransformerDiscriminator(pl.LightningModule):
    def __init__(self, num_var, tau_max, nhead=3, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.num_var = num_var
        self.tau_max = tau_max
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.transformer = torch.nn.Transformer(d_model=num_var, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=True)
        self.linear = torch.nn.Linear(tau_max*num_var, 1)
        self.save_hyperparameters()
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.transformer(x, x)
        x = self.linear(x.view(batch_size, self.tau_max*self.num_var))
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y.unsqueeze(-1).float())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y.unsqueeze(-1).float())
        self.log('val_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def backward(self, loss):
        loss.backward(retain_graph=True)



MODELS = {
    "causal": TSLinearCausal,
    "lstm": LSTMPredictor,
    "transformer": TransformerPredictor
}

DISCRIMINATORS = {
    "lstm": LSTMDiscriminator,
    "transformer": TransformerDiscriminator
}