import torch
from src.model import SimpleCNN

def test_model_output():
    model = SimpleCNN()
    dummy = torch.randn(1,3,224,224)
    out = model(dummy)
    assert out.shape == (1,2)