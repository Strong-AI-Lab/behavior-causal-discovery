
import torch
import pytorch_lightning as pl



# Dynamical Lightning module
class DynamicalPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
    
    def training_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)

        loss = torch.nn.functional.mse_loss(y_pred, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)
    
        loss = torch.nn.functional.mse_loss(y_pred, y)
        self.log('val_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)
    
        return y_pred
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
    
    def backward(self, loss):
        loss.backward(retain_graph=True)



class DynLSTMPredictor(DynamicalPredictor):
    def __init__(self, lookback, hidden_size=128, num_layers=1):
        super().__init__()
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dimensions = 3 # x, y, z

        self.input_batch_norm = torch.nn.BatchNorm1d(self.dimensions)
        self.lstm = torch.nn.LSTM(input_size=self.dimensions, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.output_batch_norm = torch.nn.BatchNorm1d(self.hidden_size)
        self.linear = torch.nn.Linear(hidden_size, self.dimensions)
        self.save_hyperparameters()

    def forward(self, x):
        x_shape = x.shape # [batch_size, lookback, dimensions]
        
        x = self.input_batch_norm(x.reshape((-1, self.dimensions)))
        x = x.reshape(x_shape)

        x, _ = self.lstm(x)

        x = self.output_batch_norm(x.reshape((-1, self.hidden_size)))
        x = self.linear(x)
        x = x.reshape(x_shape)

        return x
    

class DynMLPPredictor(DynamicalPredictor):
    def __init__(self, lookback, hidden_size=128, num_layers=1):
        super().__init__()
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dimensions = 3 # x, y, z

        self.input_batch_norm = torch.nn.BatchNorm1d(self.dimensions)
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(self.dimensions * self.lookback, self.hidden_size),
            torch.nn.ReLU()
        )

        hidden_layers = []
        for _ in range(num_layers-1):
            hidden_layers.append(torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.ReLU()
            ))
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)

        self.output_batch_norm = torch.nn.BatchNorm1d(self.hidden_size)
        self.outpout_layer = torch.nn.Linear(self.hidden_size, self.dimensions)

        self.save_hyperparameters()

    def forward(self, x):
        x_shape = x.shape # [batch_size, lookback, dimensions]
        
        x = self.batch_norm(x.reshape((-1, self.dimensions)))
        x = x.reshape(x_shape)

        x = self.input_layer(x.reshape((-1, self.dimensions * self.lookback)))

        for layer in self.hidden_layers: # Residual connections
            x = x + layer(x)

        x = self.output_batch_norm(x)
        x = self.output_layer(x)
        x = x.reshape(x_shape)

        return x
    

class DynTransformerPredictor(DynamicalPredictor):
    def __init__(self, lookback, hidden_size=192, nhead=3, num_encoder_layers=1, num_decoder_layers=1):
        super().__init__()
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dimensions = 3 # x, y, z

        self.input_batch_norm = torch.nn.BatchNorm1d(self.dimensions)
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(self.dimensions, self.hidden_size),
            torch.nn.ReLU()
        )
        self.transformer = torch.nn.Transformer(d_model=hidden_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=True)
        self.output_batch_norm = torch.nn.BatchNorm1d(self.hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, self.dimensions)
        self.save_hyperparameters()
    
    def forward(self, x):
        x_shape = x.shape # [batch_size, lookback, dimensions]
        lookback = min(self.lookback, x_shape[1])
        
        x = self.input_batch_norm(x.reshape((-1, self.dimensions)))
        x = x.reshape(x_shape)
        x = self.input_layer(x)
        
        outputs = torch.zeros_like(x)
        for i in range(1,lookback+1):
            output_i = self.transformer(x[:,:i,:], x[:,:i,:])
            outputs[:,i-1,:] = output_i[:,-1,:]
        
        outputs = self.output_batch_norm(outputs.reshape((-1, self.hidden_size)))
        outputs = self.output_layer(outputs)
        outputs = outputs.reshape(x_shape)

        return outputs



DYNAMIC_MODELS = {
    "dynamical_lstm": DynLSTMPredictor,
    "dynamical_mlp": DynMLPPredictor,
    "dynamical_transformer": DynTransformerPredictor,
}