"""Run-directory, crash-safety, and config-strictness guarantees of the runner."""

from __future__ import annotations

import json

import pytest
from pattern_recognition.experiment import run_experiment
from pattern_recognition.experiment.runner import _pipeline_params
from pattern_recognition.experiment.schema import ExperimentConfig
from pydantic import ValidationError


def _config(tmp_path, **overrides):
    cfg = {
        "name": "robust",
        "seed": 0,
        "device": "cpu",
        "data": {
            "pipeline": "SyntheticBinary",
            "params": {"n_train": 8, "n_val": 8, "n_times": 64, "n_channels": 1},
        },
        "model": {"name": "BaseCNN", "params": {"input_feat_dim": 64, "n_channels": 1}},
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 4,
            "num_epochs": 1,
            "step_size": 1,
            "gamma": 0.5,
            "save_model": True,
        },
        "output_dir": str(tmp_path),
    }
    cfg.update(overrides)
    return cfg


def test_same_second_runs_do_not_overwrite_each_other(tmp_path):
    first = run_experiment(_config(tmp_path))
    second = run_experiment(_config(tmp_path, seed=999))

    assert first != second
    assert first.is_dir() and second.is_dir()
    assert json.loads((first / "run_meta.json").read_text())["seed"] == 0
    assert json.loads((second / "run_meta.json").read_text())["seed"] == 999
    for run_dir in (first, second):
        assert (run_dir / "metrics.json").is_file()
        assert (run_dir / "model.pt").is_file()


def test_failed_training_still_leaves_the_run_record(tmp_path, monkeypatch):
    import pattern_recognition.experiment.runner as runner

    monkeypatch.setattr(
        runner,
        "train_model",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated CUDA OOM")),
    )

    with pytest.raises(RuntimeError, match="simulated CUDA OOM"):
        run_experiment(_config(tmp_path))

    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, "the failed run should still have a directory"
    run_dir = run_dirs[0]

    assert (run_dir / "config.json").is_file()
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["status"] == "failed"
    assert "simulated CUDA OOM" in meta["error"]
    assert meta["finished_at"] is not None
    assert not (run_dir / "metrics.json").exists()


def test_completed_run_records_status(tmp_path):
    run_dir = run_experiment(_config(tmp_path))
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["model_mode"] == "flash_scorer"
    assert meta["finished_at"] is not None


def test_conflicting_seeds_are_rejected_not_silently_overridden():
    cfg = ExperimentConfig.model_validate(
        {
            "name": "seeds",
            "data": {"pipeline": "SyntheticBinary", "params": {"seed": 123}},
            "model": {"name": "BaseCNN", "params": {"input_feat_dim": 64}},
            "train": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "batch_size": 4,
                "num_epochs": 1,
                "step_size": 1,
                "gamma": 0.5,
            },
            "split": {"seed": 7},
        }
    )
    with pytest.raises(ValueError, match="Conflicting split seeds"):
        _pipeline_params(cfg)


def test_matching_seeds_are_accepted():
    cfg = ExperimentConfig.model_validate(
        {
            "name": "seeds",
            "data": {"pipeline": "SyntheticBinary", "params": {"seed": 7}},
            "model": {"name": "BaseCNN", "params": {"input_feat_dim": 64}},
            "train": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "batch_size": 4,
                "num_epochs": 1,
                "step_size": 1,
                "gamma": 0.5,
            },
            "split": {"seed": 7},
        }
    )
    assert _pipeline_params(cfg)["seed"] == 7


@pytest.mark.parametrize(
    "bad_key, payload",
    [
        ("sead", {"sead": 999}),
        ("num_epochs", {"num_epochs": 500}),
        ("data.pipelien", {"data": {"pipelien": "SyntheticBinary"}}),
        ("train.epochs", {"train": {"epochs": 5}}),
    ],
)
def test_misspelled_config_keys_are_rejected(bad_key, payload):
    base = {
        "name": "typo",
        "data": {"pipeline": "SyntheticBinary", "params": {}},
        "model": {"name": "BaseCNN", "params": {"input_feat_dim": 64}},
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 4,
            "num_epochs": 1,
            "step_size": 1,
            "gamma": 0.5,
        },
    }
    for key, value in payload.items():
        if isinstance(value, dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value

    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ExperimentConfig.model_validate(base)
