import json
from pathlib import Path

import joblib
import pattern_recognition.models  # noqa: F401
from pattern_recognition.experiment import run_experiment


def _svm_cfg(tmp_path: Path) -> dict:
    return {
        "name": "svm_smoke",
        "seed": 0,
        "device": "cpu",
        "data": {
            "pipeline": "SyntheticBinary",
            "params": {
                "n_train": 32,
                "n_val": 16,
                "n_channels": 1,
                "n_times": 64,
            },
        },
        "model": {
            "name": "SVM",
            "params": {"C": 1.0, "kernel": "linear", "probability": True},
        },
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 8,
            "num_epochs": 1,
            "step_size": 1,
            "gamma": 1.0,
            "save_model": True,
        },
        "output_dir": str(tmp_path),
    }


def test_svm_run_writes_joblib_not_pt(tmp_path: Path):
    run_dir = run_experiment(_svm_cfg(tmp_path))
    assert (run_dir / "model.joblib").is_file()
    assert not (run_dir / "model.pt").exists()
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["model_mode"] == "flash_scorer"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    for key in ("accuracy", "balanced_accuracy", "f1", "itr", "train_time_sec"):
        assert key in metrics
    hist = __import__("numpy").load(run_dir / "history.npz")
    assert hist["accuracy"].shape == (1,)
    clf = joblib.load(run_dir / "model.joblib")
    assert hasattr(clf, "predict")


def test_svm_rejects_selection_packets(tmp_path: Path):
    import pytest

    cfg = _svm_cfg(tmp_path)
    cfg["data"] = {
        "pipeline": "SyntheticSelectionPackets",
        "params": {
            "protocol": "bci3_rowcol",
            "n_train": 4,
            "n_val": 2,
            "n_channels": 1,
            "n_times": 64,
            "r_max": 2,
            "seed": 0,
        },
    }
    with pytest.raises(ValueError, match="SelectionPackets"):
        run_experiment(cfg)
