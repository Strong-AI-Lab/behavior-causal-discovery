
from .dynamics_model import DynamicalPredictor, DEFAULT_ACCELERATION_PENALTY, DEFAULT_ENERGY_PENALTY, DEFAULT_FRICTION_PENALTY, DEFAULT_VELOCITY_PENALTY

import torch
import torch_geometric as tg
import pytorch_lightning as pl




class DynamicalGraphPredictor(DynamicalPredictor):

    def training_step(self, batch, batch_idx):
        x, v, g, y, i = batch
        y_pred = self(coordinates=x, velocity=v, adjacency=g)
        friction = self.friction_force(v)
        y_pred = y_pred + friction

        # Model computes the output for all individuals, move them to batch level for loss computation
        y_pred = y_pred.permute(0,2,1,3).reshape(-1, y_pred.shape[1], y_pred.shape[3])
        y = y.permute(0,2,1,3).reshape(-1, y.shape[1], y.shape[3])
        x = x.permute(0,2,1,3).reshape(-1, x.shape[1], x.shape[3])
        v = v.permute(0,2,1,3).reshape(-1, v.shape[1], v.shape[3])

        loss = self.compute_losses(y_pred, y, x, v, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, v, g, y, i = batch
        y_pred = self(coordinates=x, velocity=v, adjacency=g)
        friction = self.friction_force(v)
        y_pred = y_pred + friction

        # Model computes the output for all individuals, move them to batch level for loss computation
        y_pred = y_pred.permute(0,2,1,3).reshape(-1, y_pred.shape[1], y_pred.shape[3])
        y = y.permute(0,2,1,3).reshape(-1, y.shape[1], y.shape[3])
        x = x.permute(0,2,1,3).reshape(-1, x.shape[1], x.shape[3])
        v = v.permute(0,2,1,3).reshape(-1, v.shape[1], v.shape[3])
        
        loss = self.compute_losses(y_pred, y, x, v, 'val')
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, v, g, y, i = batch
        y_pred = self(coordinates=x, velocity=v, adjacency=g)
        y_pred = y_pred + self.friction_force(v)
    
        return y_pred
    


class DynGCNPredictor(DynamicalGraphPredictor):
    def __init__(self, lookback, friction_penalty=DEFAULT_FRICTION_PENALTY, acceleration_penalty=DEFAULT_ACCELERATION_PENALTY, velocity_penalty=DEFAULT_VELOCITY_PENALTY, energy_penalty=DEFAULT_ENERGY_PENALTY, hidden_channels=64, num_layers=3, dropout=0.5):
        super(DynGCNPredictor, self).__init__(friction_penalty, acceleration_penalty, velocity_penalty, energy_penalty)
        
        self.dimensions = 3 # x, y, z
        self.lookback = lookback
        self.model = tg.nn.Sequential(
            "x, edge_index, edge_weight",
            [
                (tg.nn.GCNConv(in_channels=2*self.dimensions, out_channels=hidden_channels), "x, edge_index, edge_weight -> x"),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ] + \
            [
                (tg.nn.GCNConv(in_channels=hidden_channels, out_channels=hidden_channels), "x, edge_index, edge_weight -> x"),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ] * (num_layers-1) + \
            [torch.nn.Linear(hidden_channels, self.dimensions)],
        )
        
        self.save_hyperparameters()

    def forward(self, coordinates, velocity, adjacency):
        x = torch.cat([coordinates, velocity], dim=-1)
        x = x.view(-1, x.size(-1))

        adjacency = adjacency.view(-1, adjacency.shape[-2], adjacency.shape[-1])
        edge_index, edge_attr = tg.utils.dense_to_sparse(adjacency)

        x = self.model(x, edge_index, edge_attr)
        x = x.view(coordinates.shape[0], coordinates.shape[1], coordinates.shape[2], self.dimensions)
        return x
    

class DynGATPredictor(DynamicalGraphPredictor):
    def __init__(self, lookback, friction_penalty=DEFAULT_FRICTION_PENALTY, acceleration_penalty=DEFAULT_ACCELERATION_PENALTY, velocity_penalty=DEFAULT_VELOCITY_PENALTY, energy_penalty=DEFAULT_ENERGY_PENALTY, hidden_channels=64, num_layers=3, dropout=0.5):
        super(DynGATPredictor, self).__init__(friction_penalty, acceleration_penalty, velocity_penalty, energy_penalty)
        
        self.dimensions = 3 # x, y, z
        self.lookback = lookback
        self.model = tg.nn.Sequential(
            "x, edge_index, edge_attr",
            [
                (tg.nn.GATConv(in_channels=2*self.dimensions, out_channels=hidden_channels), "x, edge_index, edge_attr -> x"),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ] + \
            [
                (tg.nn.GATConv(in_channels=hidden_channels, out_channels=hidden_channels), "x, edge_index, edge_attr -> x"),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ] * (num_layers-1) + \
            [torch.nn.Linear(hidden_channels, self.dimensions)],
        )
        
        self.save_hyperparameters()

    def forward(self, coordinates, velocity, adjacency):
        x = torch.cat([coordinates, velocity], dim=-1)
        x = x.view(-1, x.size(-1))

        adjacency = adjacency.view(-1, adjacency.shape[-2], adjacency.shape[-1])
        edge_index, edge_attr = tg.utils.dense_to_sparse(adjacency)

        x = self.model(x, edge_index, edge_attr)
        x = x.view(coordinates.shape[0], coordinates.shape[1], coordinates.shape[2], self.dimensions)
        return x
    

class DynGATv2Predictor(DynamicalGraphPredictor):
    def __init__(self, lookback, friction_penalty=DEFAULT_FRICTION_PENALTY, acceleration_penalty=DEFAULT_ACCELERATION_PENALTY, velocity_penalty=DEFAULT_VELOCITY_PENALTY, energy_penalty=DEFAULT_ENERGY_PENALTY, hidden_channels=64, num_layers=3, dropout=0.5):
        super(DynGATv2Predictor, self).__init__(friction_penalty, acceleration_penalty, velocity_penalty, energy_penalty)
        
        self.dimensions = 3 # x, y, z
        self.lookback = lookback
        self.model = tg.nn.Sequential(
            "x, edge_index, edge_attr",
            [
                (tg.nn.GATv2Conv(in_channels=2*self.dimensions, out_channels=hidden_channels, edge_dim=1), "x, edge_index, edge_attr -> x"),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ] + \
            [
                (tg.nn.GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, edge_dim=1), "x, edge_index, edge_attr -> x"),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ] * (num_layers-1) + \
            [torch.nn.Linear(hidden_channels, self.dimensions)],
        )
        
        self.save_hyperparameters()

    def forward(self, coordinates, velocity, adjacency):
        x = torch.cat([coordinates, velocity], dim=-1)
        x = x.view(-1, x.size(-1))

        adjacency = adjacency.view(-1, adjacency.shape[-2], adjacency.shape[-1])
        edge_index, edge_attr = tg.utils.dense_to_sparse(adjacency)

        x = self.model(x, edge_index, edge_attr)
        x = x.view(coordinates.shape[0], coordinates.shape[1], coordinates.shape[2], self.dimensions)
        return x
    


GRAPH_DYNAMIC_MODELS = {
    "dynamical_gcn": DynGCNPredictor,
    "dynamical_gat": DynGATPredictor,
    "dynamical_gatv2": DynGATv2Predictor,
}