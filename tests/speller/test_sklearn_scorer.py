from pathlib import Path

import numpy as np
import pattern_recognition.models  # noqa: F401
from pattern_recognition.experiment import run_experiment
from pattern_recognition.speller.benchmark import (
    SklearnFlashScorer,
    load_flash_scorer_from_run,
    run_speller_benchmark,
)
from pattern_recognition.speller.types import Selection


def _svm_run(tmp_path: Path) -> Path:
    return run_experiment(
        {
            "name": "svm_speller",
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
    )


def test_sklearn_flash_scorer_shapes():
    from sklearn.svm import SVC

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 64))
    y = rng.integers(0, 2, size=20)
    clf = SVC(kernel="linear", probability=True).fit(X, y)
    scorer = SklearnFlashScorer(clf)
    flashes = rng.normal(size=(5, 1, 64)).astype(np.float32)
    sel = Selection(
        flashes=flashes,
        stimulus_ids=np.arange(5),
        target_char="A",
        repeat_index=np.zeros(5, dtype=np.int64),
        meta={},
    )
    scores = scorer.predict_scores(sel)
    assert scores.shape == (5,)
    assert scores.dtype == np.float32


def test_load_flash_scorer_from_svm_run(tmp_path: Path):
    run_dir = _svm_run(tmp_path)
    scorer = load_flash_scorer_from_run(run_dir, model_mode="flash_scorer")
    assert isinstance(scorer, SklearnFlashScorer)
    flashes = np.random.randn(4, 1, 64).astype(np.float32)
    sel = Selection(
        flashes=flashes,
        stimulus_ids=np.arange(4),
        target_char="A",
        repeat_index=np.zeros(4, dtype=np.int64),
        meta={},
    )
    scores = scorer.predict_scores(sel)
    assert scores.shape == (4,)


def test_speller_smoke_with_svm_run(tmp_path: Path):
    run_dir = _svm_run(tmp_path)
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
    assert Path(out).is_dir()
