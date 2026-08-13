"""BCI Competition III Dataset II Pz epoch-averaging pipeline."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import scipy.io as sio
import torch

from pattern_recognition.data.averaging import (
    build_multichannel_subject_dataset_unique,
    multichannel_to_single_channel,
    standardize_per_sample,
)
from pattern_recognition.data.bci3_prep import (
    DEFAULT_TEST_A_CHARS,
    DEFAULT_TEST_B_CHARS,
)
from pattern_recognition.data.datasets import CNNMatrixDataset
from pattern_recognition.data.p300 import P300Getter
from pattern_recognition.data.pipelines.base import DatasetBundle
from pattern_recognition.data.pipelines.registry import register_pipeline


@register_pipeline("BCI3PzEpochAverage")
class BCI3PzEpochAverage:
    """BCI Competition III Dataset II → Pz channel → MC/SC epoch averages.

    Wraps :class:`~pattern_recognition.data.p300.P300Getter` with channel
    selection and the shared averaging helpers. Requires local ``.mat`` files
    and an electrode montage file (``eloc64.loc``).

    Parameters
    ----------
    train_mat : str
        Path to training ``.mat`` (e.g. ``Subject_A_Train.mat``).
    test_mat : str, optional
        Path to test ``.mat``. When provided, used as the validation/test
        split (matching the matrix notebooks). When omitted, train is split
        with ``val_fraction``.
    eloc_path : str
        Path to ``eloc64.loc`` (or similar). Required; no placeholder montage
        is synthesized.
    channel_name : str, default ``"Pz"``
        Electrode to keep (case-insensitive; trailing dots stripped).
    n_average : int, default 10
        Epochs grouped into one multi-channel sample.
    mode : {"SC", "MC"}, default ``"SC"``
    n_channels : int, default 64
        Full montage size expected by ``P300Getter``.
    sfreq : float, default 120
    sample_size : int, default 72
        Samples per flash epoch after resampling.
    filter : bool, default True
        Band-pass + resample via ``P300Getter.filter``.
    test_chars : list[str] | str, optional
        Target characters for the test set (required when labels are absent).
        Defaults to Subject A string when ``subject`` is ``"A"``, Subject B
        when ``subject`` is ``"B"``.
    subject : str, default ``"A"``
        Used only to select default ``test_chars``.
    val_fraction : float, default 0.2
        Used only when ``test_mat`` is not provided.
    seed : int, default 0
    """

    def __init__(
        self,
        train_mat: str,
        test_mat: str | None = None,
        eloc_path: str | None = None,
        channel_name: str = "Pz",
        n_average: int = 10,
        mode: str = "SC",
        n_channels: int = 64,
        sfreq: float = 120.0,
        sample_size: int = 72,
        filter: bool = True,
        test_chars: list[str] | str | None = None,
        subject: str = "A",
        val_fraction: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.train_mat = train_mat
        self.test_mat = test_mat
        self.eloc_path = eloc_path
        self.channel_name = channel_name
        self.n_average = n_average
        self.mode = mode.upper()
        if self.mode not in {"SC", "MC"}:
            raise ValueError(f"mode must be 'SC' or 'MC', got {mode!r}")
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.sample_size = sample_size
        self.filter = filter
        self.test_chars = test_chars
        self.subject = subject.upper()
        self.val_fraction = val_fraction
        self.seed = seed

    def _resolve_eloc(self):
        if self.eloc_path is None:
            raise ValueError(
                "eloc_path is required for BCI3PzEpochAverage "
                "(path to eloc64.loc or equivalent montage file)"
            )
        eloc_path = Path(self.eloc_path).expanduser().resolve()
        if not eloc_path.is_file():
            raise FileNotFoundError(f"Electrode montage not found: {eloc_path}")
        return mne.channels.read_custom_montage(str(eloc_path))

    def _channel_index(self, eloc) -> int:
        names = [c.lower().strip(".") for c in eloc.ch_names]
        key = self.channel_name.lower().strip(".")
        try:
            return names.index(key)
        except ValueError as exc:
            raise KeyError(
                f"Channel {self.channel_name!r} not in montage names {eloc.ch_names}"
            ) from exc

    def _default_test_chars(self) -> list[str] | None:
        if self.test_chars is not None:
            if isinstance(self.test_chars, str):
                return list(self.test_chars)
            return list(self.test_chars)
        if self.subject == "A":
            return list(DEFAULT_TEST_A_CHARS)
        if self.subject == "B":
            return list(DEFAULT_TEST_B_CHARS)
        return None

    def _load_split(
        self,
        mat_path: Path,
        eloc,
        target_chars: list[str] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = sio.loadmat(str(mat_path))
        getter = P300Getter(
            raw,
            eloc,
            n_channels=self.n_channels,
            sfreq=self.sfreq,
            sample_size=self.sample_size,
            target_chars=target_chars,
        )
        getter.get_cnn_p300_dataset(filter=self.filter)
        X, y = getter.get_data()  # [n, n_channels, sample_size], [n]
        ch_idx = self._channel_index(eloc)
        # P300Getter returns torch; select Pz and standardize per epoch.
        x_ch = X[:, ch_idx, :].detach().cpu().numpy()
        x_ch = standardize_per_sample(x_ch)
        y_np = y.detach().cpu().numpy().astype(np.int64).reshape(-1)
        return (
            torch.from_numpy(x_ch).float(),
            torch.from_numpy(y_np).long(),
        )

    def _average(
        self, x: torch.Tensor, y: torch.Tensor, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X, yy = build_multichannel_subject_dataset_unique(
            x, y, n_channels=self.n_average, seed=seed
        )
        if self.mode == "SC":
            X = multichannel_to_single_channel(X)
        return X, yy

    def build(self) -> DatasetBundle:
        train_path = Path(self.train_mat).expanduser().resolve()
        if not train_path.is_file():
            raise FileNotFoundError(f"BCI3 train .mat not found: {train_path}")

        eloc = self._resolve_eloc()
        if len(eloc.ch_names) != self.n_channels:
            raise ValueError(
                f"Montage has {len(eloc.ch_names)} channels but n_channels={self.n_channels}"
            )

        x_train, y_train = self._load_split(train_path, eloc, target_chars=None)

        test_path = None
        if self.test_mat is not None:
            test_path = Path(self.test_mat).expanduser().resolve()
            if not test_path.is_file():
                raise FileNotFoundError(f"BCI3 test .mat not found: {test_path}")
            chars = self._default_test_chars()
            x_val, y_val = self._load_split(test_path, eloc, target_chars=chars)
        else:
            from sklearn.model_selection import train_test_split

            x_np = x_train.numpy()
            y_np = y_train.numpy()
            x_tr, x_va, y_tr, y_va = train_test_split(
                x_np,
                y_np,
                test_size=self.val_fraction,
                shuffle=True,
                random_state=self.seed,
                stratify=y_np if len(np.unique(y_np)) > 1 else None,
            )
            x_train = torch.from_numpy(x_tr).float()
            y_train = torch.from_numpy(y_tr).long()
            x_val = torch.from_numpy(x_va).float()
            y_val = torch.from_numpy(y_va).long()

        X_tr, y_tr = self._average(x_train, y_train, seed=self.seed)
        X_va, y_va = self._average(x_val, y_val, seed=self.seed + 1)

        train_ds = CNNMatrixDataset((X_tr, y_tr), with_target=True, num_classes=2)
        val_ds = CNNMatrixDataset((X_va, y_va), with_target=True, num_classes=2)
        return DatasetBundle(
            train=train_ds,
            val=val_ds,
            test=None,
            metadata={
                "pipeline": "BCI3PzEpochAverage",
                "train_mat": str(train_path),
                "test_mat": str(test_path) if test_path else None,
                "eloc_path": self.eloc_path,
                "channel_name": self.channel_name,
                "n_average": self.n_average,
                "mode": self.mode,
                "sample_size": self.sample_size,
                "subject": self.subject,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
            },
        )
