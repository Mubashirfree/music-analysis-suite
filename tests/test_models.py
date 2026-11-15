import torch

from offline.training.models import ECA, GenreCNN


def _params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def test_eca_keeps_shape():
    x = torch.randn(2, 64, 16, 16)
    assert ECA(64)(x).shape == x.shape


def test_cnn_output_shape():
    out = GenreCNN()(torch.randn(2, 1, 128, 130))
    assert out.shape == (2, 10)


def test_eca_adds_few_parameters():
    extra = _params(GenreCNN(use_eca=True)) - _params(GenreCNN(use_eca=False))
    assert 0 < extra < 100