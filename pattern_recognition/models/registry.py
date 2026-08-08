"""Registry for named model factory callables."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

_MODELS: dict[str, Callable[..., Any]] = {}

T = TypeVar("T", bound=Callable[..., Any])


def register_model(name: str) -> Callable[[T], T]:
    """Decorator that registers a model factory under ``name``."""

    def decorator(fn: T) -> T:
        _MODELS[name] = fn
        return fn

    return decorator


def get_model(name: str) -> Callable[..., Any]:
    """Return a registered model factory by name."""
    try:
        return _MODELS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_MODELS)) or "(none)"
        raise KeyError(
            f"Unknown model {name!r}. Available: {available}"
        ) from exc


def list_models() -> list[str]:
    """Return sorted registered model names."""
    return sorted(_MODELS)
