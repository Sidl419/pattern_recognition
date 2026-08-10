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


_ENCODER_KEYS = (
    "d_model",
    "nhead",
    "num_layers",
    "dim_feedforward",
    "dropout",
    "max_flashes",
    "max_repetitions",
    "num_stimulus_codes",
)

_SC_KEYS = ("head_mode", "include_character_head", "n_cells")


def _build_sequence_encoder(**params: Any):
    params = dict(params)
    eegnet_params = _map_n_channels(dict(params.pop("eegnet", {})))
    from pattern_recognition.models.cnn import EEGNet, P300SequenceEncoder

    eegnet = EEGNet(**eegnet_params)
    encoder_params = {k: params.pop(k) for k in _ENCODER_KEYS if k in params}
    return P300SequenceEncoder(eegnet, **encoder_params)


@register_model("ContextualTransformer")
def build_contextual_transformer(**params: Any):
    from pattern_recognition.models.cnn import ContextualTransformer

    encoder = _build_sequence_encoder(**params)
    return ContextualTransformer(encoder)


@register_model("SequenceClassifier")
def build_sequence_classifier(**params: Any):
    from pattern_recognition.models.cnn import SequenceClassifier

    params = dict(params)
    sc_params = {k: params.pop(k) for k in _SC_KEYS if k in params}
    encoder = _build_sequence_encoder(**params)
    return SequenceClassifier(encoder, **sc_params)


__all__ = [
    "build_basecnn",
    "build_contextual_transformer",
    "build_eegnet",
    "build_sequence_classifier",
    "get_model",
    "list_models",
    "register_model",
]
