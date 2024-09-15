
from model.dynamics_model import DynamicalPredictor, DEFAULT_ACCELERATION_PENALTY, DEFAULT_ENERGY_PENALTY, DEFAULT_FRICTION_PENALTY, DEFAULT_VELOCITY_PENALTY

import torch
import torch_geometric as tg



class DynamicalGraphPredictor(DynamicalPredictor):

    def training_step(self, batch : tg.data.Batch, batch_idx):
        y_pred = self(coordinates=batch.x, velocity=batch.v, adjacency_index=batch.edge_index, adjacency_attr=batch.edge_attr)

        friction = self.friction_force(batch.v)
        y_pred = y_pred + friction
        
        y_pred = y_pred.view(1, -1, y_pred.shape[-1]) # Add batch dimension for loss computation
        y = batch.a.view(1, -1, batch.a.shape[-1])
        x = batch.x.view(1, -1, batch.x.shape[-1])
        v = batch.v.view(1, -1, batch.v.shape[-1])

        loss = self.compute_losses(y_pred, y, x, v, 'train')
        return loss
    
    def validation_step(self, batch : tg.data.Batch, batch_idx):
        y_pred = self(coordinates=batch.x, velocity=batch.v, adjacency_index=batch.edge_index, adjacency_attr=batch.edge_attr)

        friction = self.friction_force(batch.v)
        y_pred = y_pred + friction
        
        y_pred = y_pred.view(1, -1, y_pred.shape[-1])  # Add batch dimension for loss computation
        y = batch.a.view(1, -1, batch.a.shape[-1])
        x = batch.x.view(1, -1, batch.x.shape[-1])
        v = batch.v.view(1, -1, batch.v.shape[-1])
        
        loss = self.compute_losses(y_pred, y, x, v, 'val')
        return loss
    
    def test_step(self, batch : tg.data.Batch, batch_idx):
        y_pred = self(coordinates=batch.x, velocity=batch.v, adjacency_index=batch.edge_index, adjacency_attr=batch.edge_attr)

        friction = self.friction_force(batch.v)
        y_pred = y_pred + friction
        
        y_pred = y_pred.view(1, -1, y_pred.shape[-1]) # Add batch dimension for loss computation
        y = batch.a.view(1, -1, batch.a.shape[-1])
        x = batch.x.view(1, -1, batch.x.shape[-1])
        v = batch.v.view(1, -1, batch.v.shape[-1])

        loss = self.compute_losses(y_pred, y, x, v, 'test')
        return loss
    
    def predict_step(self, batch : tg.data.Batch, batch_idx):
        y_pred = self(coordinates=batch.x, velocity=batch.v, adjacency_index=batch.edge_index, adjacency_attr=batch.edge_attr)
        y_pred = y_pred + self.friction_force(batch.v)
    
        return y_pred

    def forward(self, coordinates, velocity, adjacency_index, adjacency_attr):
        x = torch.cat([coordinates, velocity], dim=-1)
        x = self.model(x, adjacency_index, adjacency_attr)
        return x
    


class DynGCNPredictor(DynamicalGraphPredictor):

    @classmethod
    def add_to_parser(cls, parser):
        parser = super(DynGCNPredictor, cls).add_to_parser(parser)
        parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels in the model.')
        parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the model.')
        parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
        return parser

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
    

class DynGATPredictor(DynamicalGraphPredictor):

    @classmethod
    def add_to_parser(cls, parser):
        parser = super(DynGATPredictor, cls).add_to_parser(parser)
        parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels in the model.')
        parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the model.')
        parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
        return parser

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
    

class DynGATv2Predictor(DynamicalGraphPredictor):

    @classmethod
    def add_to_parser(cls, parser):
        parser = super(DynGATv2Predictor, cls).add_to_parser(parser)
        parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels in the model.')
        parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the model.')
        parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
        return parser

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
    


GRAPH_DYNAMIC_MODELS = {
    "dynamical_gcn": DynGCNPredictor,
    "dynamical_gat": DynGATPredictor,
    "dynamical_gatv2": DynGATv2Predictor,
}