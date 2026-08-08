"""Samara within-subject epoch-averaging pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from pattern_recognition.data.averaging import (
    build_multichannel_subject_dataset_unique,
    multichannel_to_single_channel,
)
from pattern_recognition.data.datasets import CNNMatrixDataset
from pattern_recognition.data.pipelines.base import DatasetBundle
from pattern_recognition.data.pipelines.registry import register_pipeline
from pattern_recognition.data.time_shift import load_p300_subjects


@register_pipeline("SamaraWithinSubjectAverage")
class SamaraWithinSubjectAverage:
    """Load Samara P300 ``.mat`` files and build within-subject MC/SC averages.

    Parameters
    ----------
    path : str
        Directory containing Samara ``.mat`` files (e.g. ``Samara_data/``).
    channel_idx : int, default 1
        EEG channel index (0=P4, 1=PZ, 2=P3).
    n_average : int, default 10
        Epochs grouped into one multi-channel sample.
    mode : {"SC", "MC"}, default "SC"
        ``MC`` keeps ``n_average`` channels; ``SC`` averages them to 1 channel.
    file_pattern : str, default ``S*-P300_classic.mat``
        Glob pattern relative to ``path``.
    epoch_len : int, default 250
        Truncate each raw epoch to this many samples (from onset).
    val_fraction : float, default 0.2
        Per-subject validation fraction.
    seed : int, default 0
        Split / averaging RNG seed.
    subject : str, optional
        If set, keep only this subject id; otherwise pool all subjects.
    """

    def __init__(
        self,
        path: str,
        channel_idx: int = 1,
        n_average: int = 10,
        mode: str = "SC",
        file_pattern: str = "S*-P300_classic.mat",
        epoch_len: int = 250,
        val_fraction: float = 0.2,
        seed: int = 0,
        subject: str | None = None,
    ) -> None:
        self.path = path
        self.channel_idx = channel_idx
        self.n_average = n_average
        self.mode = mode.upper()
        if self.mode not in {"SC", "MC"}:
            raise ValueError(f"mode must be 'SC' or 'MC', got {mode!r}")
        self.file_pattern = file_pattern
        self.epoch_len = epoch_len
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
                raise KeyError(
                    f"Subject {self.subject!r} not in loaded set {subjects}"
                )
            subjects = [self.subject]

        train_X_parts: list[torch.Tensor] = []
        train_y_parts: list[torch.Tensor] = []
        val_X_parts: list[torch.Tensor] = []
        val_y_parts: list[torch.Tensor] = []

        for i, subj in enumerate(subjects):
            x = np.asarray(data[subj][:, : self.epoch_len], dtype=np.float32)
            y = np.asarray(labels[subj], dtype=np.int64)
            x_tr, x_va, y_tr, y_va = train_test_split(
                x,
                y,
                test_size=self.val_fraction,
                shuffle=True,
                random_state=self.seed + i,
                stratify=y if len(np.unique(y)) > 1 else None,
            )

            X_tr, y_tr_t = build_multichannel_subject_dataset_unique(
                x_tr, y_tr, n_channels=self.n_average, seed=self.seed + i
            )
            X_va, y_va_t = build_multichannel_subject_dataset_unique(
                x_va, y_va, n_channels=self.n_average, seed=self.seed + i + 1000
            )
            if self.mode == "SC":
                X_tr = multichannel_to_single_channel(X_tr)
                X_va = multichannel_to_single_channel(X_va)

            train_X_parts.append(X_tr)
            train_y_parts.append(y_tr_t)
            val_X_parts.append(X_va)
            val_y_parts.append(y_va_t)

        X_train = torch.cat(train_X_parts, dim=0)
        y_train = torch.cat(train_y_parts, dim=0)
        X_val = torch.cat(val_X_parts, dim=0)
        y_val = torch.cat(val_y_parts, dim=0)

        train_ds = CNNMatrixDataset(
            (X_train, y_train), with_target=True, num_classes=2
        )
        val_ds = CNNMatrixDataset(
            (X_val, y_val), with_target=True, num_classes=2
        )
        return DatasetBundle(
            train=train_ds,
            val=val_ds,
            test=None,
            metadata={
                "pipeline": "SamaraWithinSubjectAverage",
                "path": str(resolved),
                "channel_idx": self.channel_idx,
                "n_average": self.n_average,
                "mode": self.mode,
                "epoch_len": self.epoch_len,
                "val_fraction": self.val_fraction,
                "seed": self.seed,
                "subjects": subjects,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
            },
        )
