"""Samara simulated selection-packet pipeline for sequence models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pattern_recognition.data.datasets import SelectionPacketDataset
from pattern_recognition.data.pipelines.base import DatasetBundle
from pattern_recognition.data.pipelines.registry import register_pipeline
from pattern_recognition.data.samara import (
    as_split_indices,
    load_samara_subjects,
    subject_epoch_splits,
)
from pattern_recognition.data.splits import subject_seed
from pattern_recognition.selections.grids import SAMARA_GRID
from pattern_recognition.selections.packing import pack_selection
from pattern_recognition.selections.simulate import simulate_samara_selections

PROTOCOL = "samara_single_flash_sim"


@register_pipeline("SamaraSelectionPackets")
class SamaraSelectionPackets:
    """Simulate Samara single-flash selections and pack train/val packets.

    Reuses :func:`~pattern_recognition.data.time_shift.load_p300_subjects`,
    :func:`~pattern_recognition.data.splits.three_way_epoch_split`, and
    :func:`~pattern_recognition.selections.simulate.simulate_samara_selections`.
    Metadata always records ``label_source: simulated``.
    """

    def __init__(
        self,
        path: str,
        phrase: str = "JUST_DO_IT",
        r_max: int = 10,
        channel_idx: int = 1,
        epoch_len: int = 250,
        file_pattern: str = "S*-P300_classic.mat",
        seed: int = 0,
        epoch_holdout: float = 0.3,
        val_fraction: float = 0.2,
        stratify: bool = True,
        subject: str | None = None,
        simulation_seed: int | None = None,
        protocol: str | None = None,
    ) -> None:
        if not phrase:
            raise ValueError("phrase must be a non-empty string")
        if protocol is not None and protocol != PROTOCOL:
            raise ValueError(
                f"SamaraSelectionPackets protocol must be {PROTOCOL!r}, got {protocol!r}"
            )
        self.protocol = PROTOCOL
        self.path = path
        self.phrase = phrase
        self.r_max = int(r_max)
        self.channel_idx = channel_idx
        self.epoch_len = epoch_len
        self.file_pattern = file_pattern
        self.seed = seed
        self.epoch_holdout = float(epoch_holdout)
        self.val_fraction = float(val_fraction)
        self.stratify = stratify
        self.subject = subject
        self.simulation_seed = simulation_seed

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

        sim_seed = self.seed if self.simulation_seed is None else self.simulation_seed
        train_packets: list[dict] = []
        val_packets: list[dict] = []

        for subj in subjects:
            x = np.asarray(data[subj][:, : self.epoch_len], dtype=np.float32)
            y = np.asarray(labels[subj], dtype=np.int64)
            epochs = x[:, np.newaxis, :]
            train_idx, val_idx = splits[subj]["train"], splits[subj]["val"]

            train_sels = simulate_samara_selections(
                epochs,
                y,
                train_idx,
                phrase=self.phrase,
                r_max=self.r_max,
                seed=subject_seed(sim_seed, subj),
                grid=SAMARA_GRID,
            )
            val_sels = simulate_samara_selections(
                epochs,
                y,
                val_idx,
                phrase=self.phrase,
                r_max=self.r_max,
                seed=subject_seed(sim_seed + 1000, subj),
                grid=SAMARA_GRID,
            )
            for sel in train_sels:
                sel.meta = {
                    **sel.meta,
                    "subject": subj,
                    "label_source": "simulated",
                    "split": "train",
                }
            for sel in val_sels:
                sel.meta = {
                    **sel.meta,
                    "subject": subj,
                    "label_source": "simulated",
                    "split": "val",
                }
            train_packets.extend(
                pack_selection(sel, protocol=PROTOCOL) for sel in train_sels
            )
            val_packets.extend(
                pack_selection(sel, protocol=PROTOCOL) for sel in val_sels
            )

        train = SelectionPacketDataset(train_packets)
        val = SelectionPacketDataset(val_packets)
        return DatasetBundle(
            train=train,
            val=val,
            test=None,
            metadata={
                "pipeline": "SamaraSelectionPackets",
                "protocol": self.protocol,
                "label_source": "simulated",
                "path": str(resolved),
                "phrase": self.phrase,
                "r_max": self.r_max,
                "channel_idx": self.channel_idx,
                "epoch_len": self.epoch_len,
                "file_pattern": self.file_pattern,
                "seed": self.seed,
                "simulation_seed": sim_seed,
                "epoch_holdout": self.epoch_holdout,
                "val_fraction": self.val_fraction,
                "stratify": self.stratify,
                "subjects": subjects,
                "split_indices": as_split_indices(splits),
                "n_train": len(train),
                "n_val": len(val),
            },
        )
