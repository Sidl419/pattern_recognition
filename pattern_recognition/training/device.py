from __future__ import annotations

import torch


def resolve_device(requested: str) -> tuple[str, torch.device]:
    requested = requested.strip().lower()
    if requested == "cpu":
        return "cpu", torch.device("cpu")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0", torch.device("cuda:0")
        return "cpu", torch.device("cpu")
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{requested}' but CUDA is unavailable. "
                "Use 'cpu' or 'auto'."
            )
        if requested == "cuda":
            return "cuda:0", torch.device("cuda:0")
        idx = int(requested.split(":", 1)[1])
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {requested} but only {torch.cuda.device_count()} CUDA device(s)."
            )
        return f"cuda:{idx}", torch.device(f"cuda:{idx}")
    raise ValueError(f"Unknown device '{requested}'. Expected auto|cpu|cuda|cuda:N.")
