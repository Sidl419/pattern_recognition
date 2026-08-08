"""Models package — register CNN factories at import time."""

from __future__ import annotations

from typing import Any

from pattern_recognition.models.registry import (
    get_model,
    list_models,
    register_model,
)


def _map_n_channels(params: dict[str, Any]) -> dict[str, Any]:
    """Map config alias ``n_channels`` → ``in_channels`` when needed."""
    params = dict(params)
    if "n_channels" in params and "in_channels" not in params:
        params["in_channels"] = params.pop("n_channels")
    else:
        params.pop("n_channels", None)
    return params


@register_model("EEGNet")
def build_eegnet(**params: Any):
    params = _map_n_channels(params)
    from pattern_recognition.models.cnn import EEGNet

    return EEGNet(**params)


@register_model("BaseCNN")
def build_basecnn(**params: Any):
    # BaseCNN already accepts ``n_channels``; drop unused ``in_channels`` alias.
    params = dict(params)
    if "in_channels" in params and "n_channels" not in params:
        params["n_channels"] = params.pop("in_channels")
    else:
        params.pop("in_channels", None)
    from pattern_recognition.models.cnn import BaseCNN

    return BaseCNN(**params)


__all__ = [
    "build_basecnn",
    "build_eegnet",
    "get_model",
    "list_models",
    "register_model",
]
