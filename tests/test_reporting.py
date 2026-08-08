import json

import numpy as np

from pattern_recognition.experiment import run_experiment
from pattern_recognition.reporting import (
    compare_runs,
    load_run,
    metrics_table,
    plot_training_curves,
)


def _cfg(name, tmp_path):
    return {
        "name": name,
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


def _write_fake_run(tmp_path, name, *, mode=None, n_average=None):
    """Minimal run-dir artifacts for reporting-only tests."""
    run_dir = tmp_path / name
    run_dir.mkdir()
    params = {"n_train": 8, "n_val": 4}
    if mode is not None:
        params["mode"] = mode
    if n_average is not None:
        params["n_average"] = n_average
    config = {
        "name": name,
        "device": "cpu",
        "data": {"pipeline": "FakePipeline", "params": params},
        "model": {"name": "FakeModel", "params": {}},
        "train": {},
        "output_dir": str(tmp_path),
    }
    meta = {"name": name, "device_requested": "cpu", "device_resolved": "cpu"}
    metrics = {
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "f1": 0.5,
        "itr": 0.1,
        "train_time_sec": 1.0,
        "device_requested": "cpu",
        "device_resolved": "cpu",
    }
    (run_dir / "config.json").write_text(json.dumps(config) + "\n")
    (run_dir / "run_meta.json").write_text(json.dumps(meta) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics) + "\n")
    np.savez(
        run_dir / "history.npz",
        loss=np.asarray([1.0, 0.5]),
        accuracy=np.asarray([0.4, 0.5]),
    )
    return run_dir


def test_metrics_table_two_runs(tmp_path):
    d1 = run_experiment(_cfg("run_a", tmp_path))
    d2 = run_experiment(_cfg("run_b", tmp_path))
    run = load_run(d1)
    assert "accuracy" in run.metrics
    assert run.config["name"] == "run_a"
    assert "val_loss" in run.history or "loss" in run.history
    table = metrics_table([d1, d2])
    assert len(table) == 2
    assert "name" in table.columns and "device_resolved" in table.columns
    for col in (
        "accuracy",
        "balanced_accuracy",
        "f1",
        "itr",
        "train_time_sec",
        "model.name",
        "data.pipeline",
    ):
        assert col in table.columns


def test_metrics_table_includes_optional_data_params(tmp_path):
    d = _write_fake_run(tmp_path, "run_avg", mode="average", n_average=5)
    table = metrics_table([d])
    assert "mode" in table.columns
    assert "n_average" in table.columns
    assert table.iloc[0]["mode"] == "average"
    assert table.iloc[0]["n_average"] == 5


def test_plot_helpers_return_figures(tmp_path):
    d1 = _write_fake_run(tmp_path, "plot_a")
    d2 = _write_fake_run(tmp_path, "plot_b")
    run = load_run(d1)
    fig = plot_training_curves(run)
    assert fig is not None
    figs = compare_runs([d1, d2])
    assert figs is not None
