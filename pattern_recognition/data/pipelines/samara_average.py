"""Samara within-subject epoch-averaging pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pattern_recognition.data.averaging import (
    build_multichannel_subject_dataset_unique,
    multichannel_to_single_channel,
)
from pattern_recognition.data.datasets import CNNMatrixDataset
from pattern_recognition.data.pipelines.base import DatasetBundle
from pattern_recognition.data.pipelines.registry import register_pipeline
from pattern_recognition.data.samara import (
    as_split_indices,
    load_samara_subjects,
    subject_epoch_splits,
)
from pattern_recognition.data.splits import subject_seed


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
        Fraction of the *train pool* used for validation (after holdout).
    epoch_holdout : float, default 0.0
        Fraction of epochs held out for speller eval. ``0`` keeps the legacy
        train/val-only split (no speller holdout). When set (via shared
        ``split``), holdout is disjoint from train/val.
    stratify : bool, default True
        Stratify epoch splits by label 0/1.
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
        epoch_holdout: float = 0.0,
        stratify: bool = True,
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
        self.epoch_holdout = float(epoch_holdout)
        self.stratify = stratify
        self.seed = seed
        self.subject = subject

    def build(self) -> DatasetBundle:
        data, labels, subjects = load_samara_subjects(
            self.path,
            channel_idx=self.channel_idx,
            file_pattern=self.file_pattern,
            subject=self.subject,
        )
        resolved = Path(self.path).expanduser().resolve()
        splits = subject_epoch_splits(
            labels,
            subjects,
            seed=self.seed,
            epoch_holdout=self.epoch_holdout,
            val_fraction=self.val_fraction,
            stratify=self.stratify,
        )

        train_X_parts: list[torch.Tensor] = []
        train_y_parts: list[torch.Tensor] = []
        val_X_parts: list[torch.Tensor] = []
        val_y_parts: list[torch.Tensor] = []

        for subj in subjects:
            x = np.asarray(data[subj][:, : self.epoch_len], dtype=np.float32)
            y = np.asarray(labels[subj], dtype=np.int64)
            subj_seed = subject_seed(self.seed, subj)
            train_idx, val_idx = splits[subj]["train"], splits[subj]["val"]
            x_tr, y_tr = x[train_idx], y[train_idx]
            x_va, y_va = x[val_idx], y[val_idx]

            X_tr, y_tr_t = build_multichannel_subject_dataset_unique(
                x_tr, y_tr, n_channels=self.n_average, seed=subj_seed
            )
            X_va, y_va_t = build_multichannel_subject_dataset_unique(
                x_va, y_va, n_channels=self.n_average, seed=subj_seed + 1000
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

        train_ds = CNNMatrixDataset((X_train, y_train), with_target=True, num_classes=2)
        val_ds = CNNMatrixDataset((X_val, y_val), with_target=True, num_classes=2)
        metadata: dict = {
            "pipeline": "SamaraWithinSubjectAverage",
            "path": str(resolved),
            "channel_idx": self.channel_idx,
            "n_average": self.n_average,
            "mode": self.mode,
            "epoch_len": self.epoch_len,
            "val_fraction": self.val_fraction,
            "epoch_holdout": self.epoch_holdout,
            "stratify": self.stratify,
            "seed": self.seed,
            "subjects": subjects,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
        }
        if self.epoch_holdout > 0.0:
            metadata["split_indices"] = as_split_indices(splits)
        return DatasetBundle(
            train=train_ds,
            val=val_ds,
            test=None,
            metadata=metadata,
        )
