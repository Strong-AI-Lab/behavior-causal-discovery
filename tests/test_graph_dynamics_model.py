
import pytest
import torch

from model.graph_dynamics_model import DynGCNPredictor, DynGATPredictor, DynGATv2Predictor


@pytest.mark.parametrize("model_class", [DynGCNPredictor, DynGATPredictor, DynGATv2Predictor])
class TestGraphDynamicsModel:
    @pytest.fixture
    def model(self, model_class):
        return model_class(lookback=4)

    def test_forward(self, model):
        x = torch.arange(144, dtype=torch.float32).view(2,4,6,3) # Load all individuals' coordinates
        v = torch.arange(144, dtype=torch.float32).view(2,4,6,3)
        a = torch.arange(288, dtype=torch.float32).view(2,4,6,6)
        y = model(x, velocity=v, adjacency=a)
        assert y.shape == (2, 4, 6, 3)