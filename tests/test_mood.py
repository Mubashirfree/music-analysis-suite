import numpy as np
import torch

from offline.evaluation.mood_metrics import quadrant_from_va, regression_metrics
from offline.training.mood_net import MultitaskMLP


def test_multitask_output_shapes():
    model = MultitaskMLP()
    va, logits = model(torch.randn(5, 768))
    assert va.shape == (5, 2) and logits.shape == (5, 4)
    assert float(va.detach().min()) >= 0.0 and float(va.detach().max()) <= 1.0


def test_quadrant_from_va():
    va = np.array([[0.8, 0.8], [0.2, 0.8], [0.2, 0.2], [0.8, 0.2]])
    assert quadrant_from_va(va).tolist() == [0, 1, 2, 3]


def test_perfect_prediction_has_zero_rmse():
    y = np.array([[0.1, 0.9], [0.4, 0.5], [0.7, 0.2]])
    assert regression_metrics(y, y)["valence"]["rmse"] == 0.0