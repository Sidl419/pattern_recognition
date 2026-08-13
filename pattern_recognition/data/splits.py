"""Epoch-level train / val / speller-holdout splits."""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.model_selection import train_test_split

# Keep seeds well inside sklearn's ``random_state`` range (< 2**32) so callers
# can still offset them (e.g. ``+ 1000`` for a second stream) without wrapping.
_SEED_MODULUS = 2**31


def subject_seed(base_seed: int, subject: str) -> int:
    """Derive a per-subject split seed from the subject id, not its position.

    Enumerating subjects and using ``base_seed + i`` makes the split depend on
    which subjects happen to be loaded: the same subject gets a different
    split when trained alone versus pooled, and anything recomputing the split
    later (the speller) silently disagrees with training. Hashing the id keeps
    a subject's split identical however the cohort is sliced or ordered.

    ``hashlib`` rather than ``hash()`` — the latter is salted per process.
    """
    digest = hashlib.blake2b(subject.encode("utf-8"), digest_size=8).digest()
    return (int(base_seed) + int.from_bytes(digest, "big")) % _SEED_MODULUS


def stratified_epoch_holdout(
    y: np.ndarray,
    epoch_holdout: float,
    seed: int,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Split epoch indices into train-pool and eval (speller holdout) pools."""
    if epoch_holdout <= 0.0:
        indices = np.arange(len(y))
        return indices, np.array([], dtype=int)
    if epoch_holdout >= 1.0:
        raise ValueError(f"epoch_holdout must be in (0, 1), got {epoch_holdout}")
    indices = np.arange(len(y))
    stratify_labels = y if stratify and len(np.unique(y)) > 1 else None
    train_idx, eval_idx = train_test_split(
        indices,
        test_size=epoch_holdout,
        random_state=seed,
        stratify=stratify_labels,
    )
    return train_idx, eval_idx


def three_way_epoch_split(
    y: np.ndarray,
    *,
    epoch_holdout: float,
    val_fraction: float,
    seed: int,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold out speller eval epochs, then split the remainder into train/val.

    Returns ``(train_idx, val_idx, eval_idx)`` — all pairwise disjoint and
    covering ``0..len(y)-1``.
    """
    train_pool_idx, eval_idx = stratified_epoch_holdout(
        y, epoch_holdout=epoch_holdout, seed=seed, stratify=stratify
    )
    if len(train_pool_idx) == 0:
        raise ValueError("train pool is empty after epoch holdout")
    if val_fraction <= 0.0:
        return train_pool_idx, np.array([], dtype=int), eval_idx
    if val_fraction >= 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    y_pool = y[train_pool_idx]
    stratify_labels = y_pool if stratify and len(np.unique(y_pool)) > 1 else None
    train_local, val_local = train_test_split(
        np.arange(len(train_pool_idx)),
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify_labels,
    )
    return (
        train_pool_idx[train_local],
        train_pool_idx[val_local],
        eval_idx,
    )
