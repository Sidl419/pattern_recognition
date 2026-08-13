"""Shared Samara loading / splitting helpers used by every Samara pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from pattern_recognition.data.samara import (
    as_split_indices,
    load_samara_subjects,
    split_subject_epochs,
    subject_epoch_splits,
)
from pattern_recognition.data.splits import three_way_epoch_split

SUBJECTS = ["S0201", "S0601", "S0701"]


def _labels(n: int = 200, n_pos: int = 40, seed: int = 0) -> np.ndarray:
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n - n_pos, dtype=int)])
    np.random.default_rng(seed).shuffle(y)
    return y


@pytest.fixture
def fake_samara(monkeypatch, tmp_path):
    data = {s: np.zeros((200, 500), dtype=np.float32) for s in SUBJECTS}
    labels = {s: _labels(seed=i) for i, s in enumerate(SUBJECTS)}
    monkeypatch.setattr(
        "pattern_recognition.data.samara.load_p300_subjects",
        lambda *a, **k: (data, labels),
    )
    return tmp_path, data, labels


def test_loads_all_subjects_sorted(fake_samara):
    path, _, _ = fake_samara
    _, _, subject_ids = load_samara_subjects(path)
    assert subject_ids == sorted(SUBJECTS)


def test_single_subject_filter(fake_samara):
    path, _, _ = fake_samara
    _, _, subject_ids = load_samara_subjects(path, subject="S0601")
    assert subject_ids == ["S0601"]


def test_explicit_subject_list_preserves_caller_order(fake_samara):
    path, _, _ = fake_samara
    _, _, subject_ids = load_samara_subjects(path, subjects=["S0701", "S0201"])
    assert subject_ids == ["S0701", "S0201"]


def test_unknown_subject_is_rejected(fake_samara):
    path, _, _ = fake_samara
    with pytest.raises(KeyError, match="S9999"):
        load_samara_subjects(path, subject="S9999")
    with pytest.raises(KeyError, match="S9999"):
        load_samara_subjects(path, subjects=["S0201", "S9999"])


def test_subject_and_subjects_are_mutually_exclusive(fake_samara):
    path, _, _ = fake_samara
    with pytest.raises(ValueError, match="not both"):
        load_samara_subjects(path, subject="S0201", subjects=["S0601"])


def test_missing_directory_and_empty_match(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError, match="path not found"):
        load_samara_subjects(tmp_path / "nope")

    monkeypatch.setattr(
        "pattern_recognition.data.samara.load_p300_subjects",
        lambda *a, **k: ({}, {}),
    )
    with pytest.raises(FileNotFoundError, match="No files matching"):
        load_samara_subjects(tmp_path)


def test_split_subject_epochs_matches_three_way():
    y = _labels()
    kwargs = dict(epoch_holdout=0.3, val_fraction=0.2, stratify=True)
    got = split_subject_epochs(y, seed=0, **kwargs)
    expected = three_way_epoch_split(y, seed=0, **kwargs)
    for a, b in zip(got, expected, strict=True):
        np.testing.assert_array_equal(a, b)


def test_zero_holdout_keeps_train_val_only():
    y = _labels()
    train, val, ev = split_subject_epochs(
        y, seed=0, epoch_holdout=0.0, val_fraction=0.2, stratify=True
    )
    assert len(ev) == 0
    assert set(train) | set(val) == set(range(len(y)))


def test_per_subject_splits_differ_but_are_reproducible():
    labels = {s: _labels(seed=i) for i, s in enumerate(SUBJECTS)}
    first = subject_epoch_splits(
        labels, SUBJECTS, seed=0, epoch_holdout=0.3, val_fraction=0.2, stratify=True
    )
    second = subject_epoch_splits(
        labels, SUBJECTS, seed=0, epoch_holdout=0.3, val_fraction=0.2, stratify=True
    )
    for subj in SUBJECTS:
        assert np.array_equal(first[subj]["eval"], second[subj]["eval"])

    # A subset request yields the identical split for the subjects it contains.
    subset = subject_epoch_splits(
        labels, ["S0701"], seed=0, epoch_holdout=0.3, val_fraction=0.2, stratify=True
    )
    assert np.array_equal(subset["S0701"]["eval"], first["S0701"]["eval"])


def test_as_split_indices_is_json_ready():
    labels = {s: _labels() for s in SUBJECTS}
    splits = subject_epoch_splits(
        labels, SUBJECTS, seed=0, epoch_holdout=0.3, val_fraction=0.2, stratify=True
    )
    payload = as_split_indices(splits)
    assert set(payload) == set(SUBJECTS)
    for parts in payload.values():
        assert set(parts) == {"train", "val", "eval"}
        assert all(isinstance(i, int) for i in parts["eval"])
