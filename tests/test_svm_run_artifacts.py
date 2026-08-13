import json
from pathlib import Path

import joblib
import numpy as np
import pattern_recognition.models  # noqa: F401
from pattern_recognition.experiment import run_experiment
from pattern_recognition.selections.types import Selection
from pattern_recognition.speller.benchmark import (
    SklearnFlashScorer,
    load_flash_scorer_from_run,
    run_speller_benchmark,
)


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


def test_svm_run_checkpoint_loads_and_spells(tmp_path: Path):
    """One synthetic SVM run covers artifacts, Brier, scorer load, and speller smoke."""
    run_dir = run_experiment(_svm_cfg(tmp_path))
    assert (run_dir / "model.joblib").is_file()
    assert not (run_dir / "model.pt").exists()
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["model_mode"] == "flash_scorer"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    for key in (
        "accuracy",
        "balanced_accuracy",
        "f1",
        "itr",
        "brier",
        "train_time_sec",
    ):
        assert key in metrics
    assert np.isfinite(metrics["brier"])
    assert 0.0 <= metrics["brier"] <= 2.0
    hist = np.load(run_dir / "history.npz")
    assert hist["accuracy"].shape == (1,)
    clf = joblib.load(run_dir / "model.joblib")
    assert hasattr(clf, "predict")

    scorer = load_flash_scorer_from_run(run_dir, model_mode="flash_scorer")
    assert isinstance(scorer, SklearnFlashScorer)
    flashes = np.random.default_rng(0).normal(size=(4, 1, 64)).astype(np.float32)
    sel = Selection(
        flashes=flashes,
        stimulus_ids=np.arange(4),
        target_char="A",
        repeat_index=np.zeros(4, dtype=np.int64),
        meta={},
    )
    scores = scorer.predict_scores(sel)
    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))

    out = run_speller_benchmark(
        {
            "tag": "svm_syn",
            "model_mode": "flash_scorer",
            "protocol": "bci3_rowcol",
            "subject_mode": "within_subject",
            "repetitions": [1, 2],
            "run_dir": str(run_dir),
            "use_synthetic": True,
            "protocol_params": {"flash_shape": [1, 64]},
            "plots": False,
        }
    )
    speller = json.loads((out / "speller_metrics.json").read_text())
    assert speller["acc_vs_repeats"]
    assert all(np.isfinite(row["char_acc"]) for row in speller["acc_vs_repeats"])
