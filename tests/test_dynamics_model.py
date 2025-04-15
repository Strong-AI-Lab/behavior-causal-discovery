
import pytest
import torch

from model.dynamics_model import DynLSTMPredictor, DynMLPPredictor, DynTransformerPredictor, DynVariationalPredictor, DynVariationalLSTMPredictor, DynVariationalMLPPredictor, DynVariationalTransformerPredictor


@pytest.mark.parametrize("model_class", [DynLSTMPredictor, DynMLPPredictor, DynTransformerPredictor])
class TestDynamicsModel:
    @pytest.fixture
    def model_no_velocity(self, model_class):
        return model_class(lookback=4, include_velocity=False)
    
    @pytest.fixture
    def model_velocity(self, model_class):
        return model_class(lookback=4, include_velocity=True)

    def test_forward(self, model_no_velocity):
        x = torch.arange(24, dtype=torch.float32).view(2,4,3) # 2 samples, 4 timesteps, 3 features (fixed in current implementation)
        y = model_no_velocity(x)
        assert y.shape == (2, 4, 3)

    def test_forward_with_velocity(self, model_velocity):
        x = torch.arange(24, dtype=torch.float32).view(2,4,3)
        v = torch.arange(24, dtype=torch.float32).view(2,4,3)
        y = model_velocity(x, velocity=v)
        assert y.shape == (2, 4, 3)


@pytest.mark.parametrize("model_class", [DynVariationalLSTMPredictor, DynVariationalMLPPredictor, DynVariationalTransformerPredictor])
class TestVariationalDynamicsModel:
    @pytest.fixture
    def model(self, model_class):
        return model_class(lookback=4)

    def test_forward_with_velocity(self, model):
        x = torch.arange(24, dtype=torch.float32).view(2,4,3)
        v = torch.arange(24, dtype=torch.float32).view(2,4,3)
        y = model(x, velocity=v)
        assert y.shape == (2, 4, 3)