import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from pattern_recognition.reporting import (
    load_speller_tag,
    plot_speller_acc_vs_repeats,
    plot_speller_itr_vs_repeats,
)
from pattern_recognition.speller.benchmark import OracleFlashScorer, run_speller_benchmark
from pattern_recognition.speller.protocols import get_protocol


def _write_speller_artifacts(speller_dir: Path) -> None:
    speller_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "acc_vs_repeats": [
            {"r": 1, "char_acc": 0.5, "itr": 10.0},
            {"r": 2, "char_acc": 0.75, "itr": 15.0},
            {"r": 5, "char_acc": 1.0, "itr": 20.0},
        ],
        "n_selections": 2,
        "repetitions": [1, 2, 5],
    }
    (speller_dir / "speller_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
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


def test_load_speller_tag_reads_metrics(tmp_path):
    run_dir = tmp_path / "binary_run"
    speller_dir = run_dir / "speller" / "demo"
    _write_speller_artifacts(speller_dir)

    loaded = load_speller_tag(run_dir, "demo")

    assert loaded["n_selections"] == 2
    assert loaded["repetitions"] == [1, 2, 5]
    assert len(loaded["acc_vs_repeats"]) == 3


def test_load_speller_tag_missing_raises(tmp_path):
    run_dir = tmp_path / "binary_run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_speller_tag(run_dir, "missing")


def test_plot_speller_acc_vs_repeats_returns_figure(tmp_path):
    speller_dir = tmp_path / "speller" / "demo"
    _write_speller_artifacts(speller_dir)

    fig = plot_speller_acc_vs_repeats(speller_dir)

    assert fig is not None
    ax = fig.axes[0]
    assert ax.get_xlabel() == "repetitions (r)"
    assert ax.get_ylabel() == "char accuracy"
    plt.close(fig)


def test_plot_speller_itr_vs_repeats_returns_figure(tmp_path):
    speller_dir = tmp_path / "speller" / "demo"
    _write_speller_artifacts(speller_dir)

    fig = plot_speller_itr_vs_repeats(speller_dir)

    assert fig is not None
    ax = fig.axes[0]
    assert ax.get_xlabel() == "repetitions (r)"
    assert ax.get_ylabel() == "ITR (bits/min)"
    plt.close(fig)


def _stub_binary_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "name": "smoke_binary",
                "seed": 0,
                "split": {
                    "seed": 0,
                    "epoch_holdout": 0.3,
                    "stratify": True,
                    "val_fraction": 0.2,
                },
            }
        )
        + "\n"
    )


def test_benchmark_writes_plots_when_enabled(tmp_path):
    run_dir = tmp_path / "binary_run"
    _stub_binary_run(run_dir)
    proto = get_protocol("samara_single_flash_sim")
    oracle = OracleFlashScorer(proto.grid)
    config = {
        "tag": "plots_on",
        "model_mode": "flash_scorer",
        "protocol": "samara_single_flash_sim",
        "subject_mode": "within_subject",
        "repetitions": [1, 2],
        "run_dir": str(run_dir),
        "plots": True,
        "split": {
            "seed": 0,
            "epoch_holdout": 0.3,
            "stratify": True,
            "val_fraction": 0.2,
        },
        "simulation": {"seed": 0, "phrase": "JU"},
        "use_synthetic": True,
    }

    out = run_speller_benchmark(config, scores_provider=oracle)

    plots_dir = out / "plots"
    assert (plots_dir / "acc_vs_repeats.png").is_file()
    assert (plots_dir / "itr_vs_repeats.png").is_file()


def test_benchmark_skips_plots_when_disabled(tmp_path):
    run_dir = tmp_path / "binary_run"
    _stub_binary_run(run_dir)
    proto = get_protocol("samara_single_flash_sim")
    oracle = OracleFlashScorer(proto.grid)
    config = {
        "tag": "plots_off",
        "model_mode": "flash_scorer",
        "protocol": "samara_single_flash_sim",
        "subject_mode": "within_subject",
        "repetitions": [1, 2],
        "run_dir": str(run_dir),
        "plots": False,
        "split": {
            "seed": 0,
            "epoch_holdout": 0.3,
            "stratify": True,
            "val_fraction": 0.2,
        },
        "simulation": {"seed": 0, "phrase": "JU"},
        "use_synthetic": True,
    }

    out = run_speller_benchmark(config, scores_provider=oracle)

    assert not (out / "plots").exists()
