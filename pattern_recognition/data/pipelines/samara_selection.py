"""Samara simulated selection-packet pipeline for sequence models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pattern_recognition.data.datasets import SelectionPacketDataset
from pattern_recognition.data.pipelines.base import DatasetBundle
from pattern_recognition.data.pipelines.registry import register_pipeline

PROTOCOL = "samara_single_flash_sim"


@register_pipeline("SamaraSelectionPackets")
class SamaraSelectionPackets:
    """Simulate Samara single-flash selections and pack train/val packets.

    Reuses :func:`~pattern_recognition.data.time_shift.load_p300_subjects`,
    :func:`~pattern_recognition.data.splits.three_way_epoch_split`, and
    :func:`~pattern_recognition.speller.simulate.simulate_samara_selections`.
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
    ) -> None:
        if not phrase:
            raise ValueError("phrase must be a non-empty string")
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
        # Lazy: simulate/packing pull in speller package init.
        from pattern_recognition.data.splits import three_way_epoch_split
        from pattern_recognition.data.time_shift import load_p300_subjects
        from pattern_recognition.speller.grids import SAMARA_GRID
        from pattern_recognition.speller.packing import pack_selection
        from pattern_recognition.speller.simulate import simulate_samara_selections

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

        sim_seed = self.seed if self.simulation_seed is None else self.simulation_seed
        train_packets: list[dict] = []
        val_packets: list[dict] = []
        split_indices: dict[str, dict[str, list[int]]] = {}

        for i, subj in enumerate(subjects):
            x = np.asarray(data[subj][:, : self.epoch_len], dtype=np.float32)
            y = np.asarray(labels[subj], dtype=np.int64)
            epochs = x[:, np.newaxis, :]
            subj_seed = self.seed + i

            if self.epoch_holdout > 0.0:
                train_idx, val_idx, eval_idx = three_way_epoch_split(
                    y,
                    epoch_holdout=self.epoch_holdout,
                    val_fraction=self.val_fraction,
                    seed=subj_seed,
                    stratify=self.stratify,
                )
            else:
                from sklearn.model_selection import train_test_split

                indices = np.arange(len(y))
                stratify_labels = y if self.stratify and len(np.unique(y)) > 1 else None
                train_idx, val_idx = train_test_split(
                    indices,
                    test_size=self.val_fraction,
                    random_state=subj_seed,
                    stratify=stratify_labels,
                )
                eval_idx = np.array([], dtype=int)

            split_indices[subj] = {
                "train": train_idx.astype(int).tolist(),
                "val": val_idx.astype(int).tolist(),
                "eval": eval_idx.astype(int).tolist(),
            }

            train_sels = simulate_samara_selections(
                epochs,
                y,
                train_idx,
                phrase=self.phrase,
                r_max=self.r_max,
                seed=sim_seed + i,
                grid=SAMARA_GRID,
            )
            val_sels = simulate_samara_selections(
                epochs,
                y,
                val_idx,
                phrase=self.phrase,
                r_max=self.r_max,
                seed=sim_seed + 1000 + i,
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
                "protocol": PROTOCOL,
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
                "split_indices": split_indices,
                "n_train": len(train),
                "n_val": len(val),
            },
        )
