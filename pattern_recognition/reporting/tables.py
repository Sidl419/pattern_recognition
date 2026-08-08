"""Build comparison tables from saved run directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pattern_recognition.reporting.load import load_run

_METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "itr",
    "train_time_sec",
)

_OPTIONAL_DATA_PARAMS = ("mode", "n_average")


def _row_from_run(run_dir: str | Path) -> dict[str, Any]:
    run = load_run(run_dir)
    config = run.config
    metrics = run.metrics
    meta = run.meta

    data = config.get("data") or {}
    model = config.get("model") or {}
    params = data.get("params") or {}

    row: dict[str, Any] = {
        "name": config.get("name", meta.get("name")),
        "device_resolved": metrics.get(
            "device_resolved", meta.get("device_resolved")
        ),
        "model.name": model.get("name"),
        "data.pipeline": data.get("pipeline"),
    }
    for key in _METRIC_KEYS:
        if key in metrics:
            row[key] = metrics[key]

    for key in _OPTIONAL_DATA_PARAMS:
        if key in params:
            row[key] = params[key]

    return row


def metrics_table(run_dirs: list[str | Path]) -> pd.DataFrame:
    """One row per run with metrics and identifying config fields."""
    rows = [_row_from_run(path) for path in run_dirs]
    return pd.DataFrame(rows)
