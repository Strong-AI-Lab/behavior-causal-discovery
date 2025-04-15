
import pytest
import torch

from model.graph_dynamics_model import DynGCNPredictor, DynGATPredictor, DynGATv2Predictor


@pytest.mark.parametrize("model_class", [DynGCNPredictor, DynGATPredictor, DynGATv2Predictor])
class TestGraphDynamicsModel:
    @pytest.fixture
    def model(self, model_class):
        return model_class(lookback=4)

    def test_forward(self, model):
        x = torch.arange(144, dtype=torch.float32).view(48,3) # Load all individuals' coordinates
        v = torch.arange(144, dtype=torch.float32).view(48,3)
        adj_index = torch.arange(44, dtype=torch.int64).view(2,22)
        adj_attr = torch.arange(22, dtype=torch.float32).view(22,)
        y = model(x, velocity=v, adjacency_index=adj_index, adjacency_attr=adj_attr)
        assert y.shape == (48, 3)