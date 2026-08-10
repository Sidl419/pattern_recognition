"""Speller protocol registry and implementations."""

# Import concrete protocols so @register_protocol runs at import time.
from pattern_recognition.speller.protocols import (
    bci3_rowcol,  # noqa: F401
    samara_single_flash_sim,  # noqa: F401
)
from pattern_recognition.speller.protocols.base import (
    SpellerProtocol,
    get_protocol,
    register_protocol,
)

__all__ = [
    "SpellerProtocol",
    "get_protocol",
    "register_protocol",
]
