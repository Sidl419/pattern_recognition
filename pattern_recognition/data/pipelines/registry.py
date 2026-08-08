"""Registry for named data pipeline classes."""

from __future__ import annotations

from typing import Callable, TypeVar

_PIPELINES: dict[str, type] = {}

T = TypeVar("T")


def register_pipeline(name: str) -> Callable[[T], T]:
    """Decorator that registers a pipeline class under ``name``."""

    def decorator(cls: T) -> T:
        _PIPELINES[name] = cls  # type: ignore[assignment]
        return cls

    return decorator


def get_pipeline(name: str) -> type:
    """Return a registered pipeline class by name."""
    try:
        return _PIPELINES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PIPELINES)) or "(none)"
        raise KeyError(
            f"Unknown pipeline {name!r}. Available: {available}"
        ) from exc


def list_pipelines() -> list[str]:
    """Return sorted registered pipeline names."""
    return sorted(_PIPELINES)
