"""
Time-shifted epoch generation for P300 BCI experiments.

Uses long epochs (500 data points = 2 s at 250 Hz) from Samara_data to
create multi-channel samples where each channel is a **full-length** window
(``epoch_len`` data points, default 250 = 1 s) shifted forward in time
relative to the stimulus onset.

Channel layout (shift_ms=100, epoch_len=250, fs=250):
    Ch 0 : epoch[  0 : 250]   — original 0–1000 ms window
    Ch 1 : epoch[ 25 : 275]   — shifted +100 ms
    Ch 2 : epoch[ 50 : 300]   — shifted +200 ms
    …
    Ch N-1 : epoch[(N-1)*25 : (N-1)*25 + 250]

Constraint: the last channel must fit within the raw epoch and should not
overlap with the next stimulus.  With 500-sample raw epochs and
epoch_len=250:  (N-1)*shift_samples + epoch_len ≤ 500  →  N ≤ 11
(at shift=100 ms).  In practice choose N so that the last channel ends
before the next stimulus (ISI-dependent).

Example (numpy)
---------------
>>> data, labels = load_p300_subjects("Samara_data/")
>>> X, y = build_timeshifted_dataset(
...     data["S1901"], labels["S1901"], shift_ms=100, n_channels=3
... )
>>> X.shape  # (6759, 3, 250)  — each channel is full 250 data points

Example (torch, inside notebook)
---------------------------------
>>> X_t, y_t = to_torch(X, y)
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Tuple, Union

import numpy as np
import scipy.io as sio

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _to_numpy(x) -> np.ndarray:
    """Convert input to numpy array (handles torch tensors transparently)."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_p300_subjects(
    raw_data_path: str,
    channel_idx: int = 1,
    standardize: bool = True,
    eps: float = 1e-8,
    file_pattern: str = "S*-P300_classic.mat",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load .mat files with EEG epochs (Samara_data format, 500-point epochs).

    Parameters
    ----------
    raw_data_path : str
        Directory with .mat files (e.g. ``Samara_data/``).
    channel_idx : int, default 1
        EEG channel to extract (0=P4, **1=PZ**, 2=P3).
    standardize : bool, default True
        Z-score each epoch (zero mean, unit variance).
    eps : float
        Floor for std during standardization.
    file_pattern : str, default "S*-P300_classic.mat"
        Glob pattern for .mat files (e.g. ``"S*-Aperture_B.mat"``).

    Returns
    -------
    data : dict[str, ndarray[n_trials, raw_epoch_len]]
        Full-length epochs (typically 500 data points).
    labels : dict[str, ndarray[n_trials]]
    """
    data: Dict[str, np.ndarray] = {}
    labels: Dict[str, np.ndarray] = {}

    pattern = os.path.join(raw_data_path, file_pattern)
    for fpath in sorted(glob.glob(pattern)):
        mat = sio.loadmat(fpath)
        subj = str(mat["subj"][0])
        epochs = mat["epochs"]  # (N, 3, 500)
        y = mat["labels"].ravel().astype(np.int64)

        x = epochs[:, channel_idx, :].astype(np.float32)  # (N, 500)

        if standardize:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            x = (x - mean) / np.maximum(std, eps)

        data[subj] = x
        labels[subj] = y

    return data, labels


# ---------------------------------------------------------------------------
# Time-shifted multi-channel builder
# ---------------------------------------------------------------------------


def build_timeshifted_dataset(
    data: Union[np.ndarray, "torch.Tensor"],
    labels: Union[np.ndarray, "torch.Tensor"],
    shift_ms: float = 100.0,
    n_channels: int = 3,
    epoch_len: int = 250,
    fs: float = 250.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create multi-channel data with full-length time-shifted windows.

    Each channel is ``epoch_len`` data points long.  Channel *i* starts at
    sample ``i * shift_samples``.

    Parameters
    ----------
    data : array-like [n_trials, raw_epoch_len]
        Raw epochs (e.g. 500 data points from Samara_data).
    labels : array-like [n_trials]
    shift_ms : float, default 100.0
        Shift between consecutive channels (ms).
    n_channels : int, default 3
        Number of shifted channels.
    epoch_len : int, default 250
        Length of each channel window (data points).  250 @ 250 Hz = 1 s.
    fs : float, default 250.0
        Sampling frequency (Hz).

    Returns
    -------
    X : ndarray [n_trials, n_channels, epoch_len]
    y : ndarray [n_trials]
    """
    data_np = _to_numpy(data)
    labels_np = _to_numpy(labels).copy()

    shift_samples = int(round(shift_ms * fs / 1000.0))
    n_trials, raw_len = data_np.shape

    last_start = (n_channels - 1) * shift_samples
    needed = last_start + epoch_len

    if needed > raw_len:
        raise ValueError(
            f"Need {needed} data points per epoch, but raw epochs have only "
            f"{raw_len}.  With epoch_len={epoch_len}, shift={shift_samples} "
            f"samples and {n_channels} channels, reduce n_channels or "
            f"shift_ms.  Max channels = {(raw_len - epoch_len) // shift_samples + 1}."
        )

    # Build index array: (n_channels, epoch_len)
    offsets = np.arange(n_channels) * shift_samples  # (n_channels,)
    base = np.arange(epoch_len)  # (epoch_len,)
    indices = offsets[:, None] + base[None, :]  # (n_channels, epoch_len)

    X = data_np[:, indices]  # (n_trials, n_channels, epoch_len)
    return X, labels_np


def build_timeshifted_dataset_sc(
    data: Union[np.ndarray, "torch.Tensor"],
    labels: Union[np.ndarray, "torch.Tensor"],
    shift_ms: float = 100.0,
    n_channels: int = 3,
    epoch_len: int = 250,
    fs: float = 250.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Like :func:`build_timeshifted_dataset` but averages channels → (n, 1, epoch_len)."""
    X_mc, y = build_timeshifted_dataset(
        data, labels, shift_ms, n_channels, epoch_len, fs
    )
    X_sc = X_mc.mean(axis=1, keepdims=True)
    return X_sc, y


# ---------------------------------------------------------------------------
# Torch helper
# ---------------------------------------------------------------------------


def to_torch(X: np.ndarray, y: np.ndarray):
    """Convert numpy arrays to torch tensors (float32 / int64)."""
    if not _HAS_TORCH:
        raise ImportError("torch is required for to_torch()")
    return torch.from_numpy(X).float(), torch.from_numpy(y).long()


# ---------------------------------------------------------------------------
# Info / diagnostics
# ---------------------------------------------------------------------------


def time_shift_info(
    raw_epoch_len: int = 500,
    epoch_len: int = 250,
    fs: float = 250.0,
    shift_ms: float = 100.0,
    n_channels: int = 3,
) -> dict:
    """Compute and return shift parameters for logging.

    >>> time_shift_info(500, 250, 250, 100, 3)
    {'shift_samples': 25, 'epoch_len': 250, 'last_channel_end': 300, ...}
    """
    shift_samples = int(round(shift_ms * fs / 1000.0))
    last_start = (n_channels - 1) * shift_samples
    last_end = last_start + epoch_len
    max_channels = (raw_epoch_len - epoch_len) // shift_samples + 1

    return {
        "shift_samples": shift_samples,
        "epoch_len": epoch_len,
        "epoch_ms": epoch_len / fs * 1000.0,
        "last_channel_start_sample": last_start,
        "last_channel_start_ms": last_start / fs * 1000.0,
        "last_channel_end_sample": last_end,
        "last_channel_end_ms": last_end / fs * 1000.0,
        "fits_in_raw": last_end <= raw_epoch_len,
        "max_channels": max_channels,
    }
