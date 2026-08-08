"""Base types for data pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class DatasetBundle:
    """Train/val/test datasets produced by a data pipeline."""

    train: Any  # torch Dataset or (X, y)
    val: Any
    test: Any | None = None
    metadata: dict = field(default_factory=dict)


class DataPipeline(Protocol):
    """Protocol for config-driven dataset builders."""

    def build(self) -> DatasetBundle: ...
