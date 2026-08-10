import json
from pathlib import Path

import numpy as np
import pytest

import pattern_recognition.models  # noqa: F401 — register models
from pattern_recognition.experiment import run_experiment
from pattern_recognition.speller.benchmark import load_flash_scorer_from_run
from pattern_recognition.speller.types import Selection


def _binary_run(tmp_path: Path) -> Path:
    cfg = {
        "name": "scorer_smoke",
        "seed": 0,
        "device": "cpu",
        "data": {
            "pipeline": "SyntheticBinary",
            "params": {
                "n_train": 16,
                "n_val": 8,
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
            "save_model": True,
        },
        "output_dir": str(tmp_path),
    }
    return run_experiment(cfg)


def test_load_flash_scorer_predicts_scores(tmp_path):
    run_dir = _binary_run(tmp_path)
    scorer = load_flash_scorer_from_run(run_dir, model_mode="flash_scorer")

    flashes = np.random.randn(5, 1, 64).astype(np.float32)
    selection = Selection(
        flashes=flashes,
        stimulus_ids=np.arange(5),
        target_char="A",
        repeat_index=np.zeros(5, dtype=np.int64),
        meta={},
    )
    scores = scorer.predict_scores(selection)

    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))


def test_load_flash_scorer_missing_artifacts(tmp_path):
    run_dir = _binary_run(tmp_path)
    (run_dir / "model.pt").unlink()
    with pytest.raises(FileNotFoundError, match="model checkpoint"):
        load_flash_scorer_from_run(run_dir, model_mode="flash_scorer")

    empty = tmp_path / "empty_run"
    empty.mkdir()
    (empty / "model.pt").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="config"):
        load_flash_scorer_from_run(empty, model_mode="flash_scorer")


def test_load_flash_scorer_selection_classifier_not_implemented(tmp_path):
    run_dir = _binary_run(tmp_path)

    with pytest.raises(NotImplementedError, match="selection_classifier"):
        load_flash_scorer_from_run(run_dir, model_mode="selection_classifier")
