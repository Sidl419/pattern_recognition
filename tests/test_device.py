import pytest
import torch
from pattern_recognition.training.device import resolve_device


def test_resolve_cpu():
    resolved, dev = resolve_device("cpu")
    assert resolved == "cpu"
    assert dev.type == "cpu"


def test_resolve_auto_returns_string_and_device():
    resolved, dev = resolve_device("auto")
    assert resolved in {"cpu", "cuda", "cuda:0"} or resolved.startswith("cuda")
    assert isinstance(dev, torch.device)


def test_explicit_cuda_fails_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_device("cuda")
