import json

from pattern_recognition.data.pipelines.registry import get_pipeline
from pattern_recognition.experiment import run_experiment

REQUIRED_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "f1",
    "brier",
    "itr",
    "train_time_sec",
    "best_epoch",
    "checkpoint_metric",
    "device_requested",
    "device_resolved",
}
REQUIRED_META = {
    "seed",
    "started_at",
    "finished_at",
    "device_requested",
    "device_resolved",
    "name",
    "status",
    "checkpoint_metric",
    "best_epoch",
}


def test_run_experiment_writes_contract(tmp_path):
    cfg = {
        "name": "smoke",
        "seed": 0,
        "device": "cpu",
        "data": {
            "pipeline": "SyntheticBinary",
            "params": {"n_train": 32, "n_val": 16, "n_channels": 1, "n_times": 64},
        },
        "model": {"name": "BaseCNN", "params": {"input_feat_dim": 64, "n_channels": 1}},
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
    run_dir = run_experiment(cfg)
    assert (run_dir / "config.json").exists()
    assert (run_dir / "run_meta.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "history.npz").exists()
    assert (run_dir / "model.pt").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert REQUIRED_METRICS <= set(metrics)
    assert REQUIRED_META <= set(meta)
    assert metrics["device_resolved"] == "cpu"


def test_config_seed_injected_when_params_omit_seed(tmp_path, monkeypatch):
    captured: dict = {}
    Real = get_pipeline("SyntheticBinary")

    class Capturing(Real):
        def __init__(self, **kwargs):
            captured["kwargs"] = dict(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(
        "pattern_recognition.experiment.runner.get_pipeline",
        lambda name: Capturing,
    )
    cfg = {
        "name": "seed_inject",
        "seed": 42,
        "device": "cpu",
        "data": {
            "pipeline": "SyntheticBinary",
            "params": {"n_train": 16, "n_val": 8, "n_channels": 1, "n_times": 32},
        },
        "model": {"name": "BaseCNN", "params": {"input_feat_dim": 32, "n_channels": 1}},
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
    run_experiment(cfg)
    assert captured["kwargs"]["seed"] == 42
