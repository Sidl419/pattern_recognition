"""Samara time-shifted multi-channel pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from pattern_recognition.data.datasets import CNNMatrixDataset
from pattern_recognition.data.pipelines.base import DatasetBundle
from pattern_recognition.data.pipelines.registry import register_pipeline
from pattern_recognition.data.time_shift import (
    build_timeshifted_dataset,
    build_timeshifted_dataset_sc,
    load_p300_subjects,
)


@register_pipeline("SamaraTimeShift")
class SamaraTimeShift:
    """Load Samara long epochs and build time-shifted MC (or SC) windows.

    Parameters
    ----------
    path : str
        Directory containing Samara ``.mat`` files.
    channel_idx : int, default 1
        EEG channel index (0=P4, 1=PZ, 2=P3).
    shift_ms : float, default 100
        Temporal shift between consecutive synthetic channels (ms).
    n_channels : int, default 3
        Number of time-shifted channels.
    epoch_len : int, default 250
        Length of each shifted window (samples).
    fs : float, default 250
        Sampling rate (Hz).
    mode : {"MC", "SC"}, default "MC"
        ``SC`` averages shifted channels to a singleton channel.
    file_pattern : str, default ``S*-P300_classic.mat``
    val_fraction : float, default 0.2
    seed : int, default 0
    subject : str, optional
        If set, keep only this subject id.
    """

    def __init__(
        self,
        path: str,
        channel_idx: int = 1,
        shift_ms: float = 100.0,
        n_channels: int = 3,
        epoch_len: int = 250,
        fs: float = 250.0,
        mode: str = "MC",
        file_pattern: str = "S*-P300_classic.mat",
        val_fraction: float = 0.2,
        seed: int = 0,
        subject: str | None = None,
    ) -> None:
        self.path = path
        self.channel_idx = channel_idx
        self.shift_ms = shift_ms
        self.n_channels = n_channels
        self.epoch_len = epoch_len
        self.fs = fs
        self.mode = mode.upper()
        if self.mode not in {"SC", "MC"}:
            raise ValueError(f"mode must be 'SC' or 'MC', got {mode!r}")
        self.file_pattern = file_pattern
        self.val_fraction = val_fraction
        self.seed = seed
        self.subject = subject

    def build(self) -> DatasetBundle:
        resolved = Path(self.path).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Samara data path not found: {resolved}")

        data, labels = load_p300_subjects(
            str(resolved),
            channel_idx=self.channel_idx,
            standardize=True,
            file_pattern=self.file_pattern,
        )
        if not data:
            raise FileNotFoundError(
                f"No files matching {self.file_pattern!r} under {resolved}"
            )

        subjects = sorted(data.keys())
        if self.subject is not None:
            if self.subject not in data:
                raise KeyError(f"Subject {self.subject!r} not in loaded set {subjects}")
            subjects = [self.subject]

        builder = (
            build_timeshifted_dataset_sc
            if self.mode == "SC"
            else build_timeshifted_dataset
        )

        train_X_parts: list[torch.Tensor] = []
        train_y_parts: list[torch.Tensor] = []
        val_X_parts: list[torch.Tensor] = []
        val_y_parts: list[torch.Tensor] = []

        for i, subj in enumerate(subjects):
            x = np.asarray(data[subj], dtype=np.float32)
            y = np.asarray(labels[subj], dtype=np.int64)
            x_tr, x_va, y_tr, y_va = train_test_split(
                x,
                y,
                test_size=self.val_fraction,
                shuffle=True,
                random_state=self.seed + i,
                stratify=y if len(np.unique(y)) > 1 else None,
            )

            X_tr, y_tr_np = builder(
                x_tr,
                y_tr,
                shift_ms=self.shift_ms,
                n_channels=self.n_channels,
                epoch_len=self.epoch_len,
                fs=self.fs,
            )
            X_va, y_va_np = builder(
                x_va,
                y_va,
                shift_ms=self.shift_ms,
                n_channels=self.n_channels,
                epoch_len=self.epoch_len,
                fs=self.fs,
            )

            train_X_parts.append(torch.from_numpy(X_tr).float())
            train_y_parts.append(torch.from_numpy(y_tr_np).long())
            val_X_parts.append(torch.from_numpy(X_va).float())
            val_y_parts.append(torch.from_numpy(y_va_np).long())

        X_train = torch.cat(train_X_parts, dim=0)
        y_train = torch.cat(train_y_parts, dim=0)
        X_val = torch.cat(val_X_parts, dim=0)
        y_val = torch.cat(val_y_parts, dim=0)

        train_ds = CNNMatrixDataset((X_train, y_train), with_target=True, num_classes=2)
        val_ds = CNNMatrixDataset((X_val, y_val), with_target=True, num_classes=2)
        return DatasetBundle(
            train=train_ds,
            val=val_ds,
            test=None,
            metadata={
                "pipeline": "SamaraTimeShift",
                "path": str(resolved),
                "channel_idx": self.channel_idx,
                "shift_ms": self.shift_ms,
                "n_channels": self.n_channels,
                "epoch_len": self.epoch_len,
                "fs": self.fs,
                "mode": self.mode,
                "val_fraction": self.val_fraction,
                "seed": self.seed,
                "subjects": subjects,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
            },
        )
