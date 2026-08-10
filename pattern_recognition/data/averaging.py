"""Epoch-averaging helpers for multi-channel / single-channel EEG datasets."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def standardize_per_sample(X: ArrayLike, eps: float = 1e-8) -> np.ndarray:
    """Z-score each sample (row) to zero mean and unit variance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_timestamps)
    eps : float
        Floor for std during standardization.
    """
    X_np = np.asarray(X, dtype=np.float32)
    mean = X_np.mean(axis=1, keepdims=True)
    std = X_np.std(axis=1, keepdims=True)
    return (X_np - mean) / np.maximum(std, eps)


def build_multichannel_subject_dataset_unique(
    data: ArrayLike,
    labels: ArrayLike,
    n_channels: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group unique same-class epochs into multi-channel samples.

    For each class ``{1, 0}``, shuffle, trim to a multiple of ``n_channels``,
    and reshape to ``[n_groups, n_channels, ...]``.

    Parameters
    ----------
    data : Tensor / ndarray [T, ...]
    labels : Tensor / ndarray [T]
    n_channels : int
        Number of epochs averaged (grouped) into one sample.
    seed : int, optional
        RNG seed for the within-class shuffle.

    Returns
    -------
    X : Tensor [T', n_channels, ...]
    y : Tensor [T']
    """
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, dtype=torch.float32)
    else:
        data = data.float()

    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    labels = labels.long().reshape(-1)

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    X_out: list[torch.Tensor] = []
    y_out: list[torch.Tensor] = []

    for cls in (1, 0):
        idx = torch.where(labels == cls)[0]
        X_cls = data[idx]
        M = len(X_cls)
        usable = (M // n_channels) * n_channels
        if usable == 0:
            raise ValueError(f"Not enough samples for class {cls}: {M} < {n_channels}")

        if generator is not None:
            perm = torch.randperm(M, generator=generator)
        else:
            perm = torch.randperm(M)
        X_cls = X_cls[perm][:usable]

        chunks = X_cls.view(usable // n_channels, n_channels, *X_cls.shape[1:])
        X_out.append(chunks)
        y_out.append(torch.full((usable // n_channels,), cls, dtype=torch.long))

    return torch.cat(X_out, dim=0), torch.cat(y_out, dim=0)


def multichannel_to_single_channel(X: ArrayLike) -> torch.Tensor:
    """Average over the channel dimension, keeping a singleton channel axis.

    Parameters
    ----------
    X : Tensor / ndarray [T, N, ...]

    Returns
    -------
    Tensor [T, 1, ...]
    """
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, dtype=torch.float32)
    return X.mean(dim=1, keepdim=True)
