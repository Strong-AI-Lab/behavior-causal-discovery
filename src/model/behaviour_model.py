
import torch
import pytorch_lightning as pl
import torch_geometric as tg




# TS Lightning module
class TSPredictor(pl.LightningModule):
    def __init__(self, masked_idxs_for_training=None):
        super().__init__()
        self.masked_idxs_for_training = masked_idxs_for_training
    
    def training_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)
        if self.masked_idxs_for_training is not None: # Remove masked variables
            y_pred = y_pred[:,:,torch.where(~torch.tensor([i in self.masked_idxs_for_training for i in range(self.num_var)]))[0]]
            y_pred = y_pred.softmax(dim=-1)
            y = y[:,:,torch.where(~torch.tensor([i in self.masked_idxs_for_training for i in range(self.num_var)]))[0]]

        loss = torch.nn.functional.binary_cross_entropy(y_pred, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)
        if self.masked_idxs_for_training is not None: # Remove masked variables
            y_pred = y_pred[:,:,torch.where(~torch.tensor([i in self.masked_idxs_for_training for i in range(self.num_var)]))[0]]
            y_pred = y_pred.softmax(dim=-1)
            y = y[:,:,torch.where(~torch.tensor([i in self.masked_idxs_for_training for i in range(self.num_var)]))[0]]
    
        loss = torch.nn.functional.binary_cross_entropy(y_pred, y)
        self.log('val_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y, i = batch
        y_pred = self(x)
        if self.masked_idxs_for_training is not None: # Remove masked variables
            y_pred = y_pred[:,:,torch.where(~torch.tensor([i in self.masked_idxs_for_training for i in range(self.num_var)]))[0]]
            y_pred = y_pred.softmax(dim=-1)
    
        return y_pred
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def backward(self, loss):
        loss.backward(retain_graph=True)



# Causal time series prediction model
class TSLinearCausal(TSPredictor):
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
        self.lookback = lookback # lookback = = size of the shifting window = tau_max + 1
        self.conv = torch.nn.Conv1d(1, num_variables, num_variables*lookback, stride=num_variables, padding=num_variables*(lookback-1), bias=False)

        if weights is not None: # weight shape is (num_variables, num_variables, lookback)
            weights = weights.transpose(1,2).reshape((num_variables, 1, num_variables*lookback)).to(torch.float32)
            self.conv.weight.data = weights

    def forward(self, x):
        batch_size = x.shape[0] # x shape is (batch_size, lookback, num_variables)
        x = x.view((batch_size,1, self.num_variables*self.lookback))
        x = self.conv(x)[:,:,:self.lookback].transpose(1,2)
        x = x.relu()

        return x



# Graph Neural Network model
class TSGNNPredictor(TSPredictor):
    def __init__(self, num_var, lookback, weights, hidden_size=128, gnn_class=tg.nn.GCNConv, masked_idxs_for_training=None):
        super().__init__(masked_idxs_for_training)
        self.num_var = num_var
        self.lookback = lookback
        self.graph = weights # shape is (num_var, num_var, lookback)
        self.hidden_size = hidden_size
        self.gnn_class = gnn_class

        self.input_layer = torch.nn.Linear(num_var, hidden_size)
        graph_layers = []
        for _ in range(lookback):
            graph_layers.append(gnn_class(hidden_size, hidden_size))
        self.graph_layers = torch.nn.ModuleList(graph_layers)
        self.output_layer = torch.nn.Linear(hidden_size, 1)

        self.save_hyperparameters()

    def forward(self, x):
        batch_size = x.shape[0]
        features = torch.nn.functional.one_hot(torch.arange(self.num_var)).reshape((1,1,self.num_var,self.num_var)).repeat(batch_size,self.lookback,1,1).to(x.device)
        features = x.unsqueeze(-1) * features
        features = self.input_layer(features)

        outputs = torch.zeros_like(features) # shape is (batch_size, lookback, num_var, hidden_size)
        for i, layer in enumerate(self.graph_layers):
            features_i = features[:,:i+1,:,:].view((batch_size, (i+1)*self.num_var, self.hidden_size))
            edges_i = torch.stack(torch.where(self.graph[:,:,:i+1].permute(2,0,1).reshape(((i+1)*self.num_var, self.num_var))), dim=0).to(x.device)

            if isinstance(layer, tg.nn.GATConv) or isinstance(layer, tg.nn.GATv2Conv): # GATConv and GATv2Conv do not support static graph (see https://github.com/pyg-team/pytorch_geometric/issues/2844 and https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html)
                data_list = [tg.data.Data(features_i[j], edges_i) for j in range(len(features_i))]
                mini_batch = tg.data.Batch.from_data_list(data_list)
                batched_features_i =  mini_batch.x
                batched_edges_i = mini_batch.edge_index
                outputs[:,i,:,:] = layer(batched_features_i, batched_edges_i).reshape(features_i.shape)[:,:self.num_var,:]
            else:
                outputs[:,i,:,:] = layer(features_i, edges_i)[:,:self.num_var,:]
        
        outputs = torch.nn.functional.leaky_relu(outputs)
        outputs = self.output_layer(outputs)
        outputs = torch.nn.functional.leaky_relu(outputs)

        return outputs.view((batch_size, self.lookback, self.num_var))


# Neural Causal Discovery model
class TSNeuralCausal(TSPredictor):
    def __init__(self, num_var, lookback, neural_model, causal_model, masked_idxs_for_training=None):
        super().__init__(masked_idxs_for_training)
        self.num_var = num_var
        self.lookback = lookback
        self.neural_model = neural_model
        
        self.causal_model = causal_model # causal model must not be biased by gradient updates
        for p in self.causal_model.parameters():
            p.requires_grad = False
        
        self.save_hyperparameters(ignore=['neural_model','causal_model'])

    def forward(self, x):
        neural_response = self.neural_model(x)
        causal_response = self.causal_model(x)

        combined_response = neural_response * causal_response # TODO: to be refactored
        return combined_response


# LSTM prediction model
class LSTMPredictor(TSPredictor):
    def __init__(self, num_var, lookback, hidden_size=128, num_layers=1, masked_idxs_for_training=None):
        super().__init__(masked_idxs_for_training)
        self.num_var = num_var
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(input_size=num_var, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, num_var)
        self.save_hyperparameters()

    def forward(self, x):
        batch_size = x.shape[0]
        x, _ = self.lstm(x)
        x = self.linear(x.reshape((batch_size*self.lookback, self.hidden_size)))
        x = x.reshape((batch_size, self.lookback, self.num_var))
        return x
    

# Transformer prediction model
class TransformerPredictor(TSPredictor):
    def __init__(self, num_var, lookback, nhead=3, num_encoder_layers=1, num_decoder_layers=1, masked_idxs_for_training=None):
        super().__init__(masked_idxs_for_training)
        self.num_var = num_var
        self.lookback = lookback
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.transformer = torch.nn.Transformer(d_model=num_var, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=True)
        self.linear = torch.nn.Linear(num_var, num_var)
        self.save_hyperparameters()
    
    def forward(self, x):
        outputs = torch.zeros_like(x)
        
        for i in range(1,self.lookback+1):
            output_i = self.transformer(x[:,:i,:], x[:,:i,:])
            output_i = output_i[:,-1,:]
            outputs[:,i-1,:] = self.linear(output_i)

        return outputs

# Wrappers for loading
class CausalGCNWrapper():
    def __new__(wrapper, *args, **kwargs):
        return CausalGCNWrapper.__call__(*args,**kwargs) # forbids instance creation and calls __call__ instead

    @staticmethod
    def load_from_checkpoint(*args, num_var=None, lookback=None, **kwargs):
        return TSGNNPredictor.load_from_checkpoint(*args, num_var=num_var, lookback=lookback, gnn_class=tg.nn.GCNConv, **kwargs)
    
    @staticmethod
    def __call__(num_var, lookback, weights, hidden_size=128, masked_idxs_for_training=None):
        return TSGNNPredictor(num_var, lookback, weights, hidden_size, gnn_class=tg.nn.GCNConv, masked_idxs_for_training=masked_idxs_for_training)
    
class CausalGATWrapper():
    def __new__(wrapper, *args, **kwargs):
        return CausalGATWrapper.__call__(*args,**kwargs) # forbids instance creation and calls __call__ instead

    @staticmethod
    def load_from_checkpoint(*args, num_var=None, lookback=None, **kwargs):
        return TSGNNPredictor.load_from_checkpoint(*args, num_var=num_var, lookback=lookback, gnn_class=tg.nn.GATConv, **kwargs)
    
    @staticmethod
    def __call__(num_var, lookback, weights, hidden_size=128, masked_idxs_for_training=None):
        return TSGNNPredictor(num_var, lookback, weights, hidden_size, gnn_class=tg.nn.GATConv, masked_idxs_for_training=masked_idxs_for_training)
    
class CausalGATv2Wrapper():
    def __new__(wrapper, *args, **kwargs):
        return CausalGATv2Wrapper.__call__(*args,**kwargs) # forbids instance creation and calls __call__ instead

    @staticmethod
    def load_from_checkpoint(*args, num_var=None, lookback=None, **kwargs):
        return TSGNNPredictor.load_from_checkpoint(*args, num_var=num_var, lookback=lookback, gnn_class=tg.nn.GATv2Conv, **kwargs)
    
    @staticmethod
    def __call__(num_var, lookback, weights, hidden_size=128, masked_idxs_for_training=None):
        return TSGNNPredictor(num_var=num_var, lookback=lookback, weights=weights, hidden_size=hidden_size, gnn_class=tg.nn.GATv2Conv, masked_idxs_for_training=masked_idxs_for_training)


class CausalLSTMWrapper():
    def __new__(wrapper, *args, **kwargs):
        return CausalLSTMWrapper.__call__(*args,**kwargs) # forbids instance creation and calls __call__ instead

    @staticmethod
    def load_from_checkpoint(*args, num_var=None, lookback=None, **kwargs):
        neural_model = LSTMPredictor(num_var, lookback)
        causal_model = TSLinearCausal(num_var, lookback)
        return TSNeuralCausal.load_from_checkpoint(*args, num_var=num_var, lookback=lookback, neural_model=neural_model, causal_model=causal_model, **kwargs)
    
    @staticmethod
    def __call__(num_var, lookback, weights=None, hidden_size=128, num_layers=1, masked_idxs_for_training=None):
        neural_model = LSTMPredictor(num_var, lookback, hidden_size, num_layers)
        causal_model = TSLinearCausal(num_var, lookback, weights)
        return TSNeuralCausal(num_var, lookback, neural_model, causal_model, masked_idxs_for_training)

class CausalTransformerWrapper():
    def __new__(wrapper, *args, **kwargs):
        return CausalTransformerWrapper.__call__(*args,**kwargs) # forbids instance creation and calls __call__ instead

    @staticmethod
    def load_from_checkpoint(*args, num_var=None, lookback=None, **kwargs):
        neural_model = TransformerPredictor(num_var, lookback)
        causal_model = TSLinearCausal(num_var, lookback)
        return TSNeuralCausal.load_from_checkpoint(*args, num_var=num_var, lookback=lookback, neural_model=neural_model, causal_model=causal_model, **kwargs)
    
    @staticmethod
    def __call__(num_var, lookback, weights=None, nhead=3, num_encoder_layers=1, num_decoder_layers=1, masked_idxs_for_training=None):
        neural_model = TransformerPredictor(num_var, lookback, nhead, num_encoder_layers, num_decoder_layers)
        causal_model = TSLinearCausal(num_var, lookback, weights)
        return TSNeuralCausal(num_var, lookback, neural_model, causal_model, masked_idxs_for_training)



# LSTM Model that discriminates between real and generated time series
class LSTMDiscriminator(pl.LightningModule):
    def __init__(self, num_var, lookback, hidden_size=128, num_layers=1):
        super().__init__()
        self.num_var = num_var
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(input_size=num_var, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = torch.nn.Linear(lookback*hidden_size, 1)
        self.save_hyperparameters()

    def forward(self, x):
        batch_size = x.shape[0]
        x, _ = self.lstm(x)
        x = self.linear(x.reshape((batch_size, self.lookback*self.hidden_size)))
        x = x.sigmoid()
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_pred, y.unsqueeze(-1).float())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_pred, y.unsqueeze(-1).float())
        self.log('val_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def backward(self, loss):
        loss.backward(retain_graph=True)


# Transformer Model that discriminates between real and generated time series
class TransformerDiscriminator(pl.LightningModule):
    def __init__(self, num_var, lookback, nhead=3, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.num_var = num_var
        self.lookback = lookback
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.transformer = torch.nn.Transformer(d_model=num_var, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=True)
        self.linear = torch.nn.Linear(lookback*num_var, 1)
        self.save_hyperparameters()
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.transformer(x, x)
        x = self.linear(x.view(batch_size, self.lookback*self.num_var))
        x = x.sigmoid()
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_pred, y.unsqueeze(-1).float())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_pred, y.unsqueeze(-1).float())
        self.log('val_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def backward(self, loss):
        loss.backward(retain_graph=True)



BEHAVIOUR_MODELS = {
    "causal": TSLinearCausal,
    "lstm": LSTMPredictor,
    "transformer": TransformerPredictor,
    "causal_gcn": CausalGCNWrapper,
    "causal_gat": CausalGATWrapper,
    "causal_gatv2": CausalGATv2Wrapper,
    "causal_transformer": CausalTransformerWrapper,
    "causal_lstm": CausalLSTMWrapper,
}

DISCRIMINATORS = {
    "lstm": LSTMDiscriminator,
    "transformer": TransformerDiscriminator,
}