"""BCI Competition III selection-packet pipeline for sequence models."""

from __future__ import annotations

from pathlib import Path

from pattern_recognition.data.datasets import SelectionPacketDataset
from pattern_recognition.data.pipelines.base import DatasetBundle
from pattern_recognition.data.pipelines.registry import register_pipeline

PROTOCOL = "bci3_rowcol"


@register_pipeline("BCI3SelectionPackets")
class BCI3SelectionPackets:
    """Pack BCI3 row×col selections into train/val selection packets.

    Uses :func:`~pattern_recognition.speller.data_loading.build_bci3_selections_from_mat`
    (same flash prep as the speller). Train chars come from ``train_mat``; val
    from ``test_mat`` when provided, otherwise a holdout fraction of train.
    """

    def __init__(
        self,
        train_mat: str,
        test_mat: str | None = None,
        eloc_path: str | None = None,
        subject: str = "A",
        channel_name: str = "Pz",
        n_channels: int = 64,
        sfreq: float = 120.0,
        sample_size: int = 72,
        filter: bool = True,
        test_chars: list[str] | str | None = None,
        max_repetitions: int | None = None,
        max_chars: int | None = None,
        val_fraction: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.train_mat = train_mat
        self.test_mat = test_mat
        self.eloc_path = eloc_path
        self.subject = subject
        self.channel_name = channel_name
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.sample_size = sample_size
        self.filter = filter
        self.test_chars = test_chars
        self.max_repetitions = max_repetitions
        self.max_chars = max_chars
        self.val_fraction = val_fraction
        self.seed = seed

    def _require_eloc(self) -> str:
        if self.eloc_path is None:
            raise ValueError(
                "eloc_path is required for BCI3SelectionPackets "
                "(path to eloc64.loc or equivalent montage file)"
            )
        eloc = Path(self.eloc_path).expanduser().resolve()
        if not eloc.is_file():
            raise FileNotFoundError(f"Electrode montage not found: {eloc}")
        return str(eloc)

    def _build_kwargs(self, *, for_test: bool) -> dict:
        kwargs: dict = {
            "eloc_path": self._require_eloc(),
            "subject": self.subject,
            "channel_name": self.channel_name,
            "n_channels": self.n_channels,
            "sfreq": self.sfreq,
            "sample_size": self.sample_size,
            "apply_filter": self.filter,
            "max_repetitions": self.max_repetitions,
            "max_chars": self.max_chars,
        }
        if for_test:
            kwargs["test_chars"] = self.test_chars
        return kwargs

    def build(self) -> DatasetBundle:
        # Lazy: data_loading / packing pull in speller package init.
        from pattern_recognition.speller.data_loading import (
            build_bci3_selections_from_mat,
        )
        from pattern_recognition.speller.packing import pack_selection

        train_path = Path(self.train_mat).expanduser().resolve()
        if not train_path.is_file():
            raise FileNotFoundError(f"BCI3 train .mat not found: {train_path}")

        train_sels = build_bci3_selections_from_mat(
            train_path, **self._build_kwargs(for_test=False)
        )

        test_path = None
        if self.test_mat is not None:
            test_path = Path(self.test_mat).expanduser().resolve()
            if not test_path.is_file():
                raise FileNotFoundError(f"BCI3 test .mat not found: {test_path}")
            val_sels = build_bci3_selections_from_mat(
                test_path, **self._build_kwargs(for_test=True)
            )
        else:
            if not 0.0 < self.val_fraction < 1.0:
                raise ValueError(
                    f"val_fraction must be in (0, 1) when test_mat is omitted, "
                    f"got {self.val_fraction}"
                )
            import numpy as np

            rng = np.random.default_rng(self.seed)
            idx = rng.permutation(len(train_sels))
            n_val = max(1, int(round(len(train_sels) * self.val_fraction)))
            n_val = min(n_val, len(train_sels) - 1) if len(train_sels) > 1 else 0
            val_idx = set(idx[:n_val].tolist())
            val_sels = [sel for i, sel in enumerate(train_sels) if i in val_idx]
            train_sels = [sel for i, sel in enumerate(train_sels) if i not in val_idx]

        train = SelectionPacketDataset(
            [pack_selection(sel, protocol=PROTOCOL) for sel in train_sels]
        )
        val = SelectionPacketDataset(
            [pack_selection(sel, protocol=PROTOCOL) for sel in val_sels]
        )
        return DatasetBundle(
            train=train,
            val=val,
            test=None,
            metadata={
                "pipeline": "BCI3SelectionPackets",
                "protocol": PROTOCOL,
                "label_source": "ground_truth",
                "train_mat": str(train_path),
                "test_mat": str(test_path) if test_path else None,
                "eloc_path": self.eloc_path,
                "subject": self.subject.upper(),
                "channel_name": self.channel_name,
                "filter": self.filter,
                "max_repetitions": self.max_repetitions,
                "max_chars": self.max_chars,
                "n_train": len(train),
                "n_val": len(val),
                "seed": self.seed,
            },
        )
