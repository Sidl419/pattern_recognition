"""Shared Samara loading and per-subject splitting.

Every Samara pipeline (epoch averaging, time shift, selection packets) and the
speller's selection loader repeat the same three steps: resolve the directory,
load the subjects, then split each subject's epochs. Keeping them here means a
change to the split contract lands in one place instead of four.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from pattern_recognition.data.splits import subject_seed, three_way_epoch_split
from pattern_recognition.data.time_shift import load_p300_subjects

DEFAULT_FILE_PATTERN = "S*-P300_classic.mat"


def load_samara_subjects(
    path: str | Path,
    *,
    channel_idx: int = 1,
    file_pattern: str = DEFAULT_FILE_PATTERN,
    subject: str | None = None,
    subjects: list[str] | None = None,
    standardize: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Resolve ``path``, load Samara ``.mat`` epochs, and pick the subject set.

    Returns ``(data, labels, subject_ids)`` where ``subject_ids`` is sorted
    unless an explicit ``subjects`` list fixes the order.
    """
    if subject is not None and subjects is not None:
        raise ValueError("pass either subject or subjects, not both")

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Samara data path not found: {resolved}")

    data, labels = load_p300_subjects(
        str(resolved),
        channel_idx=channel_idx,
        standardize=standardize,
        file_pattern=file_pattern,
    )
    if not data:
        raise FileNotFoundError(f"No files matching {file_pattern!r} under {resolved}")

    available = sorted(data.keys())
    requested = [subject] if subject is not None else subjects
    if requested is None:
        return data, labels, available

    missing = [s for s in requested if s not in data]
    if missing:
        raise KeyError(f"Subjects {missing} not in loaded set {available}")
    return data, labels, list(requested)


def split_subject_epochs(
    y: np.ndarray,
    *,
    seed: int,
    epoch_holdout: float,
    val_fraction: float,
    stratify: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one subject's epochs into ``(train, val, eval)`` index arrays.

    ``epoch_holdout <= 0`` keeps the legacy train/val-only behaviour and
    returns an empty eval array.
    """
    if epoch_holdout > 0.0:
        return three_way_epoch_split(
            y,
            epoch_holdout=epoch_holdout,
            val_fraction=val_fraction,
            seed=seed,
            stratify=stratify,
        )

    indices = np.arange(len(y))
    stratify_labels = y if stratify and len(np.unique(y)) > 1 else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify_labels,
    )
    return train_idx, val_idx, np.array([], dtype=int)


def subject_epoch_splits(
    labels: dict[str, np.ndarray],
    subject_ids: list[str],
    *,
    seed: int,
    epoch_holdout: float,
    val_fraction: float,
    stratify: bool,
) -> dict[str, dict[str, np.ndarray]]:
    """Split every subject with its own id-derived seed."""
    return {
        subj: dict(
            zip(
                ("train", "val", "eval"),
                split_subject_epochs(
                    np.asarray(labels[subj]),
                    seed=subject_seed(seed, subj),
                    epoch_holdout=epoch_holdout,
                    val_fraction=val_fraction,
                    stratify=stratify,
                ),
                strict=True,
            )
        )
        for subj in subject_ids
    }


def as_split_indices(
    splits: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, list[int]]]:
    """JSON-serialisable form for the ``split_indices.json`` run artifact."""
    return {
        subj: {name: idx.astype(int).tolist() for name, idx in parts.items()}
        for subj, parts in splits.items()
    }
