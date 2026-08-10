"""Shared helpers for speller benchmark tests (importable from test modules)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_SPLIT = {
    "seed": 0,
    "epoch_holdout": 0.3,
    "stratify": True,
    "val_fraction": 0.2,
}


def stub_binary_run(
    run_dir: Path,
    *,
    n_average: int = 1,
    device: str = "cpu",
    with_run_meta: bool = True,
    subject_mode: str = "within_subject",
    split: dict[str, Any] | None = None,
    name: str = "smoke_binary",
    **config_extra: Any,
) -> None:
    """Write a minimal binary ``config.json`` (+ optional ``run_meta.json``)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "name": name,
        "seed": 0,
        "device": device,
        "subject_mode": subject_mode,
        "split": DEFAULT_SPLIT if split is None else split,
        "data": {
            "pipeline": "SamaraWithinSubjectAverage",
            "params": {"n_average": n_average},
        },
    }
    config.update(config_extra)
    (run_dir / "config.json").write_text(json.dumps(config) + "\n")
    if with_run_meta:
        (run_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "device_requested": device,
                    "device_resolved": device,
                    "seed": 0,
                    "name": name,
                }
            )
            + "\n"
        )


def samara_smoke_config(
    run_dir: Path | str,
    *,
    tag: str = "smoke",
    plots: bool = False,
    use_synthetic: bool = True,
    repetitions: list[int] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Valid Samara speller config dict for CI / oracle smokes."""
    cfg: dict[str, Any] = {
        "tag": tag,
        "model_mode": "flash_scorer",
        "protocol": "samara_single_flash_sim",
        "subject_mode": "within_subject",
        "repetitions": repetitions if repetitions is not None else [1, 2, 5],
        "run_dir": str(run_dir),
        "split": dict(DEFAULT_SPLIT),
        "simulation": {"seed": 0, "phrase": "JU"},
        "use_synthetic": use_synthetic,
        "plots": plots,
    }
    cfg.update(overrides)
    return cfg


def write_speller_artifacts(
    speller_dir: Path,
    *,
    label_source: str = "simulated",
    early_stop: bool = False,
    n_selections: int = 2,
) -> None:
    """Write minimal speller artifact CSVs/JSON for reporting tests."""
    speller_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "acc_vs_repeats": [
            {"r": 1, "char_acc": 0.5, "itr": 10.0},
            {"r": 2, "char_acc": 0.75, "itr": 15.0},
            {"r": 5, "char_acc": 1.0, "itr": 20.0},
        ],
        "n_selections": n_selections,
        "repetitions": [1, 2, 5],
    }
    if early_stop:
        metrics["early_stop"] = {
            "char_acc_early": 0.75,
            "mean_repeats_used": 2.0,
            "repeats_used": [1, 2, 2, 3],
        }
    (speller_dir / "speller_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    (speller_dir / "meta.json").write_text(
        json.dumps({"label_source": label_source, "tag": speller_dir.name}) + "\n"
    )
    with (speller_dir / "acc_vs_repeats.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["r", "char_acc", "itr"])
        writer.writeheader()
        writer.writerows(metrics["acc_vs_repeats"])
    per_subject_rows = [
        {"subject": "S1", "r": 1, "char_acc": 0.0, "itr": 8.0},
        {"subject": "S1", "r": 2, "char_acc": 0.5, "itr": 12.0},
        {"subject": "S1", "r": 5, "char_acc": 1.0, "itr": 18.0},
        {"subject": "S2", "r": 1, "char_acc": 1.0, "itr": 12.0},
        {"subject": "S2", "r": 2, "char_acc": 1.0, "itr": 18.0},
        {"subject": "S2", "r": 5, "char_acc": 1.0, "itr": 22.0},
    ]
    with (speller_dir / "per_subject.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["subject", "r", "char_acc", "itr"])
        writer.writeheader()
        writer.writerows(per_subject_rows)
    if early_stop:
        with (speller_dir / "early_stop_repeats.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["selection_id", "repeats_used"])
            writer.writeheader()
            for i, r in enumerate(metrics["early_stop"]["repeats_used"]):
                writer.writerow({"selection_id": i, "repeats_used": r})
