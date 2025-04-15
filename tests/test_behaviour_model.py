
import pytest
import torch

from model.behaviour_model import TSLinearCausal, LSTMPredictor, TransformerPredictor, CausalGCNWrapper, CausalGATWrapper, CausalGATv2Wrapper, CausalTransformerWrapper, CausalLSTMWrapper, LSTMDiscriminator, TransformerDiscriminator


@pytest.mark.parametrize("model_class", [TSLinearCausal, LSTMPredictor, TransformerPredictor, CausalTransformerWrapper, CausalLSTMWrapper])
class TestBehaviourModel:
    @pytest.fixture
    def model(self, model_class):
        return model_class(num_variables=3, lookback=4)

    def test_forward(self, model):
        x = torch.arange(24, dtype=torch.float32).view(2,4,3)
        y = model(x)
        assert y.shape == (2, 4, 3)


@pytest.mark.parametrize("model_class", [CausalGCNWrapper, CausalGATWrapper, CausalGATv2Wrapper])
class TestGraphBehaviourModel:
    @pytest.fixture
    def model(self, model_class):
        return model_class(num_variables=3, lookback=4, graph_weights=torch.eye(3).view(3,3,1).repeat(1,1,4))

    def test_forward(self, model):
        x = torch.arange(24, dtype=torch.float32).view(2,4,3)
        y = model(x)
        assert y.shape == (2, 4, 3)


@pytest.mark.parametrize("model_class", [LSTMDiscriminator, TransformerDiscriminator])
class TestDiscriminator:
    @pytest.fixture
    def model(self, model_class):
        return model_class(num_variables=3, lookback=4)

    def test_forward(self, model):
        x = torch.arange(24, dtype=torch.float32).view(2,4,3)
        y = model(x)
        assert y.shape == (2, 1)