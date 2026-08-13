"""Metric math in the training loop, checked against sklearn on a known matrix."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pattern_recognition.training.loop import train_model, validate_model
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# TP=3, FN=1, FP=2, TN=4 over 10 samples.
TRUE = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
PRED = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]


def _onehot(labels: list[int]) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(labels), num_classes=2).float()


def _loader(true_labels: list[int], pred_labels: list[int], batch_size: int = 4):
    """Inputs *are* the model outputs, so a passthrough yields ``pred_labels``."""
    return DataLoader(
        TensorDataset(_onehot(pred_labels), _onehot(true_labels)),
        batch_size=batch_size,
    )


class FrozenPassthrough(nn.Module):
    """Returns its input unchanged, but is trainable so ``backward()`` works.

    Combined with ``lr=0`` the predictions stay fixed, which isolates the
    metric bookkeeping from any actual learning.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def test_validate_model_matches_sklearn_on_known_confusion_matrix():
    acc = validate_model(nn.Identity(), _loader(TRUE, PRED), is_binary=True)

    assert acc["Corrects"] == 7
    assert acc["Accuracy"] == pytest.approx(0.7)
    assert float(acc["Balanced Accuracy"]) == pytest.approx(
        balanced_accuracy_score(TRUE, PRED)
    )
    assert acc["F1-score"] == pytest.approx(f1_score(TRUE, PRED))
    assert acc["Min Accuracy"] <= acc["Accuracy"] <= acc["Max Accuracy"]


def test_validate_model_handles_all_negative_predictions():
    """Degenerate but common under P300 imbalance: precision/recall are 0, not NaN."""
    all_negative = [0] * len(TRUE)
    acc = validate_model(nn.Identity(), _loader(TRUE, all_negative), is_binary=True)

    assert acc["F1-score"] == pytest.approx(0.0)
    assert float(acc["Balanced Accuracy"]) == pytest.approx(0.5)
    assert acc["Accuracy"] == pytest.approx(0.6)


def test_validate_model_is_perfect_on_perfect_predictions():
    acc = validate_model(nn.Identity(), _loader(TRUE, TRUE), is_binary=True)
    assert acc["Accuracy"] == pytest.approx(1.0)
    assert acc["F1-score"] == pytest.approx(1.0)
    assert float(acc["Balanced Accuracy"]) == pytest.approx(1.0)
    assert acc["ITR"] == pytest.approx(1.0)


def test_train_model_val_metrics_match_validate_model():
    """The inlined per-epoch validation must agree with the standalone helper."""
    loaders = {"train": _loader(TRUE, PRED), "val": _loader(TRUE, PRED)}
    learning_params = {
        "lr": 0.0,  # frozen weights: predictions stay exactly PRED
        "weight_decay": 0.0,
        "step_size": 1,
        "gamma": 1.0,
        "num_epochs": 3,
        "model_type": "CNN",
    }
    val_loss, acc, elapsed = train_model(
        FrozenPassthrough(), loaders, nn.MSELoss(), learning_params, is_binary=True
    )

    assert len(val_loss) == 3
    for key in ("Accuracy", "Balanced Accuracy", "F1-score", "ITR"):
        assert len(acc[key]) == 3, f"{key} should have one entry per epoch"

    reference = validate_model(nn.Identity(), _loader(TRUE, PRED), is_binary=True)
    assert float(acc["Accuracy"][-1]) == pytest.approx(reference["Accuracy"])
    assert float(acc["F1-score"][-1]) == pytest.approx(reference["F1-score"])
    assert float(acc["Balanced Accuracy"][-1]) == pytest.approx(
        float(reference["Balanced Accuracy"])
    )
    assert elapsed >= 0.0


def test_train_model_survives_batch_size_larger_than_dataset():
    """A single short batch must not produce NaN/inf metrics."""
    loaders = {
        "train": _loader(TRUE, PRED, batch_size=64),
        "val": _loader(TRUE, PRED, batch_size=64),
    }
    _, acc, _ = train_model(
        FrozenPassthrough(),
        loaders,
        nn.MSELoss(),
        {
            "lr": 0.0,
            "weight_decay": 0.0,
            "step_size": 1,
            "gamma": 1.0,
            "num_epochs": 1,
            "model_type": "CNN",
        },
        is_binary=True,
    )
    assert np.isfinite(np.asarray(acc["Accuracy"], dtype=float)).all()
    assert np.isfinite(np.asarray(acc["F1-score"], dtype=float)).all()


def test_train_model_rejects_unknown_model_type():
    loaders = {"train": _loader(TRUE, PRED), "val": _loader(TRUE, PRED)}
    with pytest.raises(ValueError, match="no such model type"):
        train_model(
            FrozenPassthrough(),
            loaders,
            nn.MSELoss(),
            {
                "lr": 0.0,
                "weight_decay": 0.0,
                "step_size": 1,
                "gamma": 1.0,
                "num_epochs": 1,
                "model_type": "TRANSFORMER",
            },
            is_binary=True,
        )
