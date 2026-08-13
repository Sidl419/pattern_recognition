"""Brier score as the training objective and as a reported/selection metric."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pattern_recognition.losses import BrierLoss
from pattern_recognition.models import cnn
from pattern_recognition.training.checkpoint import BINARY_METRICS, BestCheckpoint
from pattern_recognition.training.metrics import brier_score
from sklearn.metrics import brier_score_loss
from torch import nn


def test_brier_is_the_default_checkpoint_metric():
    from pattern_recognition.experiment.schema import TrainConfig

    train = TrainConfig(
        lr=1e-3,
        weight_decay=0.0,
        batch_size=8,
        num_epochs=1,
        step_size=1,
        gamma=1.0,
    )
    assert train.checkpoint_metric == "brier"
    assert BestCheckpoint().requested_metric == "brier"


def test_brier_score_matches_sklearn_binary_definition():
    p_pos = np.array([0.9, 0.1, 0.8, 0.3])
    truth = np.array([1, 0, 0, 1])
    probs = np.column_stack([1 - p_pos, p_pos])
    onehot = np.column_stack([1 - truth, truth])

    # Multi-category Brier is exactly twice the single-component binary form.
    assert brier_score(probs, onehot) == pytest.approx(
        2 * brier_score_loss(truth, p_pos)
    )


def test_brier_score_endpoints():
    perfect = np.array([[0.0, 1.0], [1.0, 0.0]])
    onehot = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert brier_score(perfect, onehot) == pytest.approx(0.0)
    assert brier_score(1 - perfect, onehot) == pytest.approx(2.0)
    assert brier_score(np.full((2, 2), 0.5), onehot) == pytest.approx(0.5)


def test_brier_score_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        brier_score(np.zeros((2, 2)), np.zeros((2, 3)))


def test_brier_loss_normalises_its_own_input():
    """The loss applies the softmax, so a model cannot skip the activation."""
    logits = torch.tensor([[2.0, -1.0], [-3.0, 0.5]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    probs = BrierLoss.probabilities(logits)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2))

    expected = ((probs - targets) ** 2).mean()
    assert BrierLoss()(logits, targets) == pytest.approx(float(expected))


def test_brier_loss_keeps_the_scale_of_the_mse_it_replaces():
    """Same gradient scale as nn.MSELoss on probabilities: learning rates carry over."""
    logits = torch.randn(16, 2)
    targets = torch.nn.functional.one_hot(torch.randint(0, 2, (16,)), 2).float()
    probs = BrierLoss.probabilities(logits)
    assert BrierLoss()(logits, targets) == pytest.approx(
        float(nn.MSELoss()(probs, targets))
    )
    # ...and it is the standard Brier score divided by the class count.
    assert float(BrierLoss()(logits, targets)) == pytest.approx(
        brier_score(probs.numpy(), targets.numpy()) / 2
    )


def test_brier_loss_rejects_non_one_hot_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        BrierLoss()(torch.randn(4, 2), torch.randint(0, 2, (4,)))


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: cnn.FlexCNN(input_feat_dim=64, n_channels=1), id="FlexCNN"
        ),
        pytest.param(
            lambda: cnn.CecottiCNN(input_feat_dim=64, n_channels=1), id="CecottiCNN"
        ),
        pytest.param(
            lambda: cnn.BaseCNNAttn(input_feat_dim=64, n_channels=1), id="BaseCNNAttn"
        ),
        pytest.param(
            lambda: cnn.DeepConvNet(input_feat_dim=64, in_channels=1), id="DeepConvNet"
        ),
        pytest.param(
            lambda: cnn.BaseCNN(input_feat_dim=64, n_channels=1), id="BaseCNN"
        ),
        pytest.param(lambda: cnn.EEGNet(input_feat_dim=64, in_channels=1), id="EEGNet"),
    ],
)
def test_every_cnn_emits_logits(factory):
    """One convention across the module: no model applies its own link.

    A trailing Sigmoid/Softmax would be double-applied by BrierLoss and
    silently change the objective for that architecture only.
    """
    model = factory()
    assert not any(
        isinstance(m, (nn.Sigmoid, nn.Softmax, nn.LogSoftmax)) for m in model.modules()
    ), "output activation found; BrierLoss owns the link function"

    model.eval()
    with torch.no_grad():
        out = model(torch.randn(8, 1, 64) * 25.0)
    assert out.shape == (8, 2)
    assert not torch.allclose(out.sum(dim=1), torch.ones(8))


def test_brier_tracker_prefers_the_lowest_value():
    """Brier is lower-is-better; a higher-is-better tracker would pick epoch 0."""
    model = nn.Linear(2, 2)
    tracker = BestCheckpoint("brier").for_loop(BINARY_METRICS)
    assert tracker.metric == "brier"
    for epoch, value in enumerate([0.8, 0.3, 0.55]):
        tracker.update(epoch, value, model)
    assert tracker.restore(model) == 1
