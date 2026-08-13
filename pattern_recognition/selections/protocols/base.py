"""Registry and base contract for speller protocols."""

from __future__ import annotations

from typing import Callable, Protocol, TypeVar, runtime_checkable

import numpy as np

from pattern_recognition.selections.grids import GridSpec
from pattern_recognition.selections.types import Selection

_PROTOCOLS: dict[str, SpellerProtocol] = {}

T = TypeVar("T", bound=type)


def register_protocol(name: str) -> Callable[[T], T]:
    """Decorator that registers a protocol instance under ``name``."""

    def decorator(cls: T) -> T:
        _PROTOCOLS[name] = cls()  # type: ignore[call-arg]
        return cls

    return decorator


def get_protocol(name: str) -> SpellerProtocol:
    """Return a registered protocol instance by name."""
    try:
        return _PROTOCOLS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PROTOCOLS)) or "(none)"
        raise KeyError(f"Unknown protocol {name!r}. Available: {available}") from exc


@runtime_checkable
class SpellerProtocol(Protocol):
    """Common interface for row×col and single-flash speller paradigms."""

    name: str
    grid: GridSpec
    label_source: str
    flashes_per_repeat: int
    soa_s: float

    def decode(self, scores: np.ndarray, selection: Selection, r: int) -> str:
        """Aggregate flash scores through ``r`` repeats into a character."""

    def build_synthetic_selections(self, **kwargs) -> list[Selection]:
        """Build synthetic selections for unit tests and smoke benchmarks."""
