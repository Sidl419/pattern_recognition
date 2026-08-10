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
from pattern_recognition.data.splits import three_way_epoch_split
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

        train_X_parts: list[torch.Tensor] = []
        train_y_parts: list[torch.Tensor] = []
        val_X_parts: list[torch.Tensor] = []
        val_y_parts: list[torch.Tensor] = []
        split_indices: dict[str, dict[str, list[int]]] = {}

        for i, subj in enumerate(subjects):
            x = np.asarray(data[subj][:, : self.epoch_len], dtype=np.float32)
            y = np.asarray(labels[subj], dtype=np.int64)
            subj_seed = self.seed + i

            if self.epoch_holdout > 0.0:
                train_idx, val_idx, eval_idx = three_way_epoch_split(
                    y,
                    epoch_holdout=self.epoch_holdout,
                    val_fraction=self.val_fraction,
                    seed=subj_seed,
                    stratify=self.stratify,
                )
                x_tr, y_tr = x[train_idx], y[train_idx]
                x_va, y_va = x[val_idx], y[val_idx]
                split_indices[subj] = {
                    "train": train_idx.astype(int).tolist(),
                    "val": val_idx.astype(int).tolist(),
                    "eval": eval_idx.astype(int).tolist(),
                }
            else:
                x_tr, x_va, y_tr, y_va = train_test_split(
                    x,
                    y,
                    test_size=self.val_fraction,
                    shuffle=True,
                    random_state=subj_seed,
                    stratify=(y if self.stratify and len(np.unique(y)) > 1 else None),
                )

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
        if split_indices:
            metadata["split_indices"] = split_indices
        return DatasetBundle(
            train=train_ds,
            val=val_ds,
            test=None,
            metadata=metadata,
        )
