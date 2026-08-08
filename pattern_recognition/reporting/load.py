"""Load experiment run artifacts from disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RunArtifacts:
    path: Path
    config: dict
    meta: dict
    metrics: dict
    history: dict  # arrays from npz


def load_run(path: str | Path) -> RunArtifacts:
    """Load config, meta, metrics, and history from a run directory."""
    run_dir = Path(path)
    config = json.loads((run_dir / "config.json").read_text())
    meta = json.loads((run_dir / "run_meta.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    with np.load(run_dir / "history.npz") as npz:
        history = {key: npz[key] for key in npz.files}
    return RunArtifacts(
        path=run_dir,
        config=config,
        meta=meta,
        metrics=metrics,
        history=history,
    )
