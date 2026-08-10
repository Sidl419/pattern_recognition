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


def resolve_speller_tag_dir(run_dir: str | Path, tag: str) -> Path:
    """Resolve ``run_dir/speller/<tag>`` or the newest ``<tag>_<timestamp>`` collision.

    Prefers an exact ``speller/<tag>/`` directory when it contains
    ``speller_metrics.json``. Otherwise picks the newest directory whose name
    is ``tag`` or starts with ``tag_`` and contains metrics.
    """
    run_dir = Path(run_dir)
    speller_root = run_dir / "speller"
    exact = speller_root / tag
    metrics_name = "speller_metrics.json"
    if exact.is_dir() and (exact / metrics_name).is_file():
        return exact

    if not speller_root.is_dir():
        raise FileNotFoundError(
            f"speller metrics not found for tag {tag!r}: {exact / metrics_name}"
        )

    prefix = f"{tag}_"
    candidates = [
        path
        for path in speller_root.iterdir()
        if path.is_dir()
        and (path.name == tag or path.name.startswith(prefix))
        and (path / metrics_name).is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"speller metrics not found for tag {tag!r}: {exact / metrics_name}"
        )
    return max(candidates, key=lambda path: (path.name, path.stat().st_mtime))


def load_speller_tag(run_dir: str | Path, tag: str) -> dict:
    """Load ``speller_metrics.json`` for a speller benchmark tag under ``run_dir``.

    Resolves timestamped collision directories via :func:`resolve_speller_tag_dir`.
    """
    metrics_path = resolve_speller_tag_dir(run_dir, tag) / "speller_metrics.json"
    return json.loads(metrics_path.read_text())
