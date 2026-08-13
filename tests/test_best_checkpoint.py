"""Best-epoch selection: which weights survive and which metrics get reported."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from pattern_recognition.experiment import run_experiment
from pattern_recognition.training.checkpoint import (
    BINARY_METRICS,
    SELECTION_METRICS,
    BestCheckpoint,
    resolve_checkpoint_metric,
)
from torch import nn


class Tagged(nn.Module):
    """Model whose single weight identifies the epoch it was saved at."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


def test_tracker_keeps_the_highest_scoring_epoch():
    model = Tagged()
    tracker = BestCheckpoint("balanced_accuracy").for_loop(BINARY_METRICS)

    for epoch, score in enumerate([0.5, 0.9, 0.7, 0.6]):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        tracker.update(epoch, score, model)

    with torch.no_grad():
        model.weight.fill_(99.0)  # overfit final epoch

    assert tracker.restore(model) == 1
    assert model.weight.item() == pytest.approx(1.0)


def test_tracker_minimises_val_loss():
    model = Tagged()
    tracker = BestCheckpoint("val_loss").for_loop(BINARY_METRICS)
    for epoch, loss in enumerate([1.0, 0.2, 0.4]):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        tracker.update(epoch, loss, model)
    assert tracker.restore(model) == 1
    assert model.weight.item() == pytest.approx(1.0)


def test_last_disables_selection_and_keeps_final_weights():
    model = Tagged()
    tracker = BestCheckpoint("last").for_loop(BINARY_METRICS)
    for epoch, score in enumerate([0.1, 0.9, 0.2]):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        tracker.update(epoch, score, model)
    assert not tracker.enabled
    assert tracker.restore(model) == 2
    assert model.weight.item() == pytest.approx(2.0)


def test_a_plateau_keeps_the_most_trained_epoch():
    """Balanced accuracy pins to 0.5 while a P300 model predicts all-negative.

    Keeping the first tie would save the least-trained weights.
    """
    model = Tagged()
    tracker = BestCheckpoint("balanced_accuracy").for_loop(BINARY_METRICS)
    for epoch in range(6):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        tracker.update(epoch, 0.5, model)

    assert tracker.restore(model) == 5
    assert model.weight.item() == pytest.approx(5.0)


def test_plateau_after_a_peak_does_not_override_the_peak():
    model = Tagged()
    tracker = BestCheckpoint("accuracy").for_loop(BINARY_METRICS)
    for epoch, score in enumerate([0.5, 0.9, 0.7, 0.7]):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        tracker.update(epoch, score, model)
    assert tracker.restore(model) == 1


def test_nan_scores_never_win():
    """SequenceClassifier reports NaN balanced accuracy; it must not rank."""
    model = Tagged()
    tracker = BestCheckpoint("accuracy").for_loop(BINARY_METRICS)
    tracker.update(0, 0.4, model)
    tracker.update(1, float("nan"), model)
    tracker.update(2, None, model)
    assert tracker.restore(model) == 0


def test_metric_resolution_falls_back_only_when_undefined():
    assert resolve_checkpoint_metric("balanced_accuracy", BINARY_METRICS) == (
        "balanced_accuracy"
    )
    # SC cannot rank by balanced accuracy or F1 — both are NaN there.
    assert resolve_checkpoint_metric("balanced_accuracy", SELECTION_METRICS) == (
        "accuracy"
    )
    assert resolve_checkpoint_metric("f1", SELECTION_METRICS) == "accuracy"
    assert resolve_checkpoint_metric("val_loss", SELECTION_METRICS) == "val_loss"
    assert resolve_checkpoint_metric("last", BINARY_METRICS) == "last"


def _config(tmp_path, **train_overrides):
    train = {
        "lr": 1e-2,
        "weight_decay": 0.0,
        "batch_size": 8,
        "num_epochs": 4,
        "step_size": 1,
        "gamma": 1.0,
        "save_model": True,
    }
    train.update(train_overrides)
    return {
        "name": "ckpt",
        "seed": 0,
        "device": "cpu",
        "data": {
            "pipeline": "SyntheticBinary",
            "params": {"n_train": 32, "n_val": 16, "n_times": 64, "n_channels": 1},
        },
        "model": {"name": "BaseCNN", "params": {"input_feat_dim": 64, "n_channels": 1}},
        "train": train,
        "output_dir": str(tmp_path),
    }


def test_reported_metrics_come_from_the_selected_epoch(tmp_path):
    run_dir = run_experiment(_config(tmp_path))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    meta = json.loads((run_dir / "run_meta.json").read_text())
    history = np.load(run_dir / "history.npz")

    best = metrics["best_epoch"]
    assert meta["best_epoch"] == best
    assert meta["checkpoint_metric"] == "brier"
    assert metrics["checkpoint_metric"] == "brier"

    # The reported values are that epoch's, and it is the best epoch by design.
    assert metrics["brier"] == pytest.approx(float(history["brier"][best]))
    assert metrics["accuracy"] == pytest.approx(float(history["accuracy"][best]))
    assert metrics["balanced_accuracy"] == pytest.approx(
        float(history["balanced_accuracy"][best])
    )
    assert float(history["brier"][best]) == pytest.approx(
        float(np.min(history["brier"]))
    )
    assert 0.0 <= metrics["brier"] <= 2.0
    assert metrics["best_epoch"] == int(np.argmin(history["brier"]))


def test_last_reports_the_final_epoch(tmp_path):
    run_dir = run_experiment(_config(tmp_path, checkpoint_metric="last"))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    history = np.load(run_dir / "history.npz")

    assert metrics["best_epoch"] == len(history["accuracy"]) - 1
    assert metrics["accuracy"] == pytest.approx(float(history["accuracy"][-1]))


def test_saved_checkpoint_matches_the_reported_epoch(tmp_path):
    """The weights on disk must be the ones that produced metrics.json."""
    from pattern_recognition.models import get_model

    run_dir = run_experiment(_config(tmp_path))
    metrics = json.loads((run_dir / "metrics.json").read_text())

    model = get_model("BaseCNN")(input_feat_dim=64, n_channels=1)
    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=False))
    model.eval()

    from pattern_recognition.data.pipelines import get_pipeline
    from pattern_recognition.training.loop import validate_model
    from torch.utils.data import DataLoader

    bundle = get_pipeline("SyntheticBinary")(
        n_train=32, n_val=16, n_times=64, n_channels=1, seed=0
    ).build()
    scored = validate_model(model, DataLoader(bundle.val, batch_size=8))

    assert float(scored["Balanced Accuracy"]) == pytest.approx(
        metrics["balanced_accuracy"], abs=1e-6
    )
    assert float(scored["Brier"]) == pytest.approx(metrics["brier"], abs=1e-6)


def test_sequence_classifier_falls_back_to_accuracy(tmp_path):
    from tests.speller.helpers import synthetic_sequence_experiment_config

    cfg = synthetic_sequence_experiment_config(
        tmp_path, model_name="SequenceClassifier", protocol="bci3_rowcol"
    )
    cfg["train"]["num_epochs"] = 2
    # SC reports selection accuracy only, so balanced accuracy is undefined.
    cfg["train"]["checkpoint_metric"] = "balanced_accuracy"
    run_dir = run_experiment(cfg)
    meta = json.loads((run_dir / "run_meta.json").read_text())

    assert meta["checkpoint_metric_requested"] == "balanced_accuracy"
    assert meta["checkpoint_metric"] == "accuracy"
    assert meta["best_epoch"] in (0, 1)
