import json
import math

import numpy as np
import pytest
from pattern_recognition.experiment import run_experiment

from tests.speller.helpers import synthetic_sequence_experiment_config


def test_sequence_model_rejects_non_packet_pipeline(tmp_path):
    cfg = synthetic_sequence_experiment_config(
        tmp_path, model_name="SequenceClassifier", name="sc_mismatch"
    )
    cfg["data"] = {
        "pipeline": "SyntheticBinary",
        "params": {"n_train": 16, "n_val": 8, "n_channels": 1, "n_times": 64},
    }
    cfg["train"]["save_model"] = False
    with pytest.raises(ValueError, match="selection-packet"):
        run_experiment(cfg)


def test_binary_model_rejects_selection_packet_pipeline(tmp_path):
    cfg = {
        "name": "bin_packet_mismatch",
        "seed": 0,
        "device": "cpu",
        "data": {
            "pipeline": "SyntheticSelectionPackets",
            "params": {
                "protocol": "bci3_rowcol",
                "n_train": 4,
                "n_val": 2,
                "n_channels": 1,
                "n_times": 64,
                "r_max": 2,
            },
        },
        "model": {
            "name": "BaseCNN",
            "params": {"input_feat_dim": 64, "n_channels": 1},
        },
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 8,
            "num_epochs": 1,
            "step_size": 1,
            "gamma": 1.0,
            "save_model": False,
        },
        "output_dir": str(tmp_path),
    }
    with pytest.raises(ValueError, match="SelectionPackets"):
        run_experiment(cfg)


def test_contextual_transformer_smoke_run(tmp_path):
    run_dir = run_experiment(
        synthetic_sequence_experiment_config(
            tmp_path, model_name="ContextualTransformer", name="ct_smoke"
        )
    )
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["model_mode"] == "flash_scorer"
    assert (run_dir / "model.pt").is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "accuracy" in metrics
    assert np.isfinite(metrics["accuracy"])


def test_sequence_classifier_smoke_run(tmp_path):
    run_dir = run_experiment(
        synthetic_sequence_experiment_config(
            tmp_path, model_name="SequenceClassifier", name="sc_smoke"
        )
    )
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["model_mode"] == "selection_classifier"
    assert (run_dir / "model.pt").is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert np.isfinite(metrics["accuracy"])
    assert math.isnan(metrics["balanced_accuracy"])
    assert math.isnan(metrics["f1"])
    history = np.load(run_dir / "history.npz")
    assert history["balanced_accuracy"].shape[0] >= 1
    assert math.isnan(float(history["balanced_accuracy"][-1]))
