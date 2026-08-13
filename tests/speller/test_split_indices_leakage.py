"""Speller holdout must come from the binary run, not a recomputed split."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pattern_recognition.data.selection_loading import (
    _eval_idx_from_artifact,
    build_samara_selections_from_dir,
)
from pattern_recognition.data.splits import subject_seed, three_way_epoch_split
from pattern_recognition.selections.protocols import get_protocol
from pattern_recognition.speller.benchmark import (
    OracleFlashScorer,
    _load_recorded_eval_indices,
    run_speller_benchmark,
)

from tests.speller.helpers import samara_smoke_config, stub_binary_run

POOLED_SUBJECTS = ["S0201", "S0601", "S0701", "S1201", "S1401"]


def _imbalanced_labels(n: int = 400, n_pos: int = 40, seed: int = 0) -> np.ndarray:
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n - n_pos, dtype=int)])
    np.random.default_rng(seed).shuffle(y)
    return y


def test_position_derived_seeds_would_leak_training_epochs():
    """The original bug: seeds keyed on list position rather than subject id."""
    y = _imbalanced_labels()
    position_in_pool = POOLED_SUBJECTS.index("S0701")

    train_idx, val_idx, eval_idx = three_way_epoch_split(
        y, epoch_holdout=0.3, val_fraction=0.2, seed=0 + position_in_pool, stratify=True
    )
    train_pool = set(train_idx) | set(val_idx)

    # Evaluating that subject alone used to put it back at position 0.
    _, _, eval_at_position_zero = three_way_epoch_split(
        y, epoch_holdout=0.3, val_fraction=0.2, seed=0 + 0, stratify=True
    )
    assert train_pool & set(eval_at_position_zero), "expected the old scheme to leak"

    # Reading the recorded indices is exact whatever the position.
    recorded = _eval_idx_from_artifact({"S0701": eval_idx.tolist()}, "S0701", len(y))
    assert set(recorded) == set(eval_idx)
    assert not (train_pool & set(recorded))


def test_subject_seed_is_independent_of_cohort_and_order():
    """A subject's split must not change with who else is in the cohort."""
    pooled = {s: subject_seed(0, s) for s in POOLED_SUBJECTS}
    reversed_cohort = {s: subject_seed(0, s) for s in reversed(POOLED_SUBJECTS)}
    alone = {"S0701": subject_seed(0, "S0701")}

    assert pooled == reversed_cohort
    assert alone["S0701"] == pooled["S0701"]
    assert len(set(pooled.values())) == len(POOLED_SUBJECTS), "seeds must differ"

    # Stable across processes (hashlib, not the salted builtin hash()).
    assert subject_seed(0, "S0701") == 402336479
    assert subject_seed(1, "S0701") == 402336480

    # Usable as a sklearn random_state even after the +1000 second-stream offset.
    assert all(0 <= s + 1000 < 2**32 for s in pooled.values())


def test_recompute_fallback_now_agrees_with_training(monkeypatch, tmp_path):
    """allow_split_mismatch path: recomputation matches the pooled training split."""
    y = _imbalanced_labels(n=200, n_pos=60)
    epochs = np.random.default_rng(2).normal(size=(len(y), 250)).astype(np.float32)
    subject = "S0701"

    # Training pools every subject; S0701 is not at position 0.
    _, _, training_eval = three_way_epoch_split(
        y,
        epoch_holdout=0.3,
        val_fraction=0.2,
        seed=subject_seed(0, subject),
        stratify=True,
    )

    monkeypatch.setattr(
        "pattern_recognition.data.samara.load_p300_subjects",
        lambda *a, **k: ({subject: epochs}, {subject: y}),
    )
    selections = build_samara_selections_from_dir(
        tmp_path,
        phrase="JU",
        r_max=1,
        epoch_holdout=0.3,
        seed=0,
        subjects=[subject],
    )

    used = {i for sel in selections for i in sel.meta["epoch_indices"]}
    assert used <= set(training_eval.tolist())
    assert all(sel.meta["eval_source"] == "recomputed" for sel in selections)


def test_eval_idx_from_artifact_rejects_unusable_records():
    with pytest.raises(KeyError, match="no entry in the binary run"):
        _eval_idx_from_artifact({"S0201": [1, 2]}, "S0701", 10)

    with pytest.raises(ValueError, match="empty eval split"):
        _eval_idx_from_artifact({"S0701": []}, "S0701", 10)

    # Stale artifact / different epoch_len: indices past the loaded epochs.
    with pytest.raises(ValueError, match="only 10 epochs were loaded"):
        _eval_idx_from_artifact({"S0701": [3, 42]}, "S0701", 10)


def test_load_recorded_eval_indices_reads_run_artifact(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert _load_recorded_eval_indices(run_dir) is None

    (run_dir / "split_indices.json").write_text(
        json.dumps(
            {
                "S0201": {"train": [0, 1], "val": [2], "eval": [3, 4]},
                "S0601": {"train": [0], "val": [1], "eval": []},
            }
        )
    )
    # Subjects with an empty holdout are dropped so they fail loudly later.
    assert _load_recorded_eval_indices(run_dir) == {"S0201": [3, 4]}


def test_samara_benchmark_requires_recorded_split(tmp_path, monkeypatch):
    """Without split_indices.json the benchmark refuses rather than guessing."""
    from pattern_recognition.data import selection_loading as dl

    monkeypatch.setattr(
        dl,
        "build_samara_selections_from_dir",
        lambda *a, **k: pytest.fail("must not build selections without a split"),
    )

    run_dir = tmp_path / "binary_run"
    stub_binary_run(run_dir)
    data_path = tmp_path / "Samara_data"
    data_path.mkdir()

    cfg = samara_smoke_config(
        run_dir,
        tag="no_split_indices",
        use_synthetic=False,
        repetitions=[1],
        protocol_params={"path": str(data_path)},
    )
    with pytest.raises(FileNotFoundError, match="No split_indices.json"):
        run_speller_benchmark(
            cfg,
            scores_provider=OracleFlashScorer(
                get_protocol("samara_single_flash_sim").grid
            ),
        )


def test_recorded_indices_are_passed_through_to_the_loader(tmp_path, monkeypatch):
    from pattern_recognition.data import selection_loading as dl

    captured: dict = {}

    def fake_build(*args, **kwargs):
        captured["kwargs"] = kwargs
        raise RuntimeError("stop_after_kwargs")

    monkeypatch.setattr(dl, "build_samara_selections_from_dir", fake_build)

    run_dir = tmp_path / "binary_run"
    stub_binary_run(run_dir)
    (run_dir / "split_indices.json").write_text(
        json.dumps({"S0701": {"train": [0], "val": [1], "eval": [2, 3, 4]}})
    )
    data_path = tmp_path / "Samara_data"
    data_path.mkdir()

    cfg = samara_smoke_config(
        run_dir,
        tag="recorded",
        use_synthetic=False,
        repetitions=[1],
        protocol_params={"path": str(data_path)},
    )
    with pytest.raises(RuntimeError, match="stop_after_kwargs"):
        run_speller_benchmark(
            cfg,
            scores_provider=OracleFlashScorer(
                get_protocol("samara_single_flash_sim").grid
            ),
        )

    assert captured["kwargs"]["eval_indices"] == {"S0701": [2, 3, 4]}


def test_builder_uses_recorded_indices_over_recomputation(tmp_path, monkeypatch):
    """End-to-end on the loader: recorded indices win over the seed fallback."""
    y = _imbalanced_labels(n=200, n_pos=60)
    epochs = np.random.default_rng(1).normal(size=(len(y), 250)).astype(np.float32)
    recorded_eval = (
        np.flatnonzero(y == 1)[:40].tolist() + np.flatnonzero(y == 0)[:60].tolist()
    )

    monkeypatch.setattr(
        "pattern_recognition.data.samara.load_p300_subjects",
        lambda *a, **k: ({"S0701": epochs}, {"S0701": y}),
    )

    selections = build_samara_selections_from_dir(
        tmp_path,
        phrase="JU",
        r_max=1,
        epoch_holdout=0.3,
        seed=0,
        eval_indices={"S0701": recorded_eval},
    )

    assert selections
    used = {i for sel in selections for i in sel.meta["epoch_indices"]}
    assert used <= set(recorded_eval)
    assert all(sel.meta["eval_source"] == "split_indices" for sel in selections)
