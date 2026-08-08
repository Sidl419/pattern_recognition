"""Synthetic binary classification pipeline for tests and smoke runs."""

from __future__ import annotations

import torch

from pattern_recognition.data.datasets import CNNMatrixDataset
from pattern_recognition.data.pipelines.base import DatasetBundle
from pattern_recognition.data.pipelines.registry import register_pipeline


@register_pipeline("SyntheticBinary")
class SyntheticBinary:
    """Random binary EEG-like tensors for registry and runner smoke tests.

    Yields ``CNNMatrixDataset`` splits with class-index labels that expand to
    one-hot floats in ``__getitem__``, matching the CNN ``train_model`` path.
    """

    def __init__(
        self,
        n_train: int = 32,
        n_val: int = 16,
        n_test: int | None = None,
        n_channels: int = 1,
        n_times: int = 64,
        seed: int = 0,
    ) -> None:
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.n_channels = n_channels
        self.n_times = n_times
        self.seed = seed

    def _make_split(self, n: int, generator: torch.Generator) -> CNNMatrixDataset:
        x = torch.randn(n, self.n_channels, self.n_times, generator=generator)
        y = torch.randint(0, 2, (n,), generator=generator)
        return CNNMatrixDataset((x, y), with_target=True, num_classes=2)

    def build(self) -> DatasetBundle:
        generator = torch.Generator().manual_seed(self.seed)
        train = self._make_split(self.n_train, generator)
        val = self._make_split(self.n_val, generator)
        test = None
        if self.n_test is not None:
            test = self._make_split(self.n_test, generator)
        return DatasetBundle(
            train=train,
            val=val,
            test=test,
            metadata={
                "pipeline": "SyntheticBinary",
                "n_train": self.n_train,
                "n_val": self.n_val,
                "n_test": self.n_test,
                "n_channels": self.n_channels,
                "n_times": self.n_times,
                "seed": self.seed,
            },
        )
