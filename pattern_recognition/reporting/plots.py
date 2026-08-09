"""Plot helpers that read saved run artifacts only."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pattern_recognition.reporting.load import RunArtifacts, load_run

_LOSS_KEYS = ("loss", "val_loss", "train_loss")


def _loss_key(history: dict) -> str | None:
    for key in _LOSS_KEYS:
        if key in history:
            return key
    return None


def plot_training_curves(run: RunArtifacts, ax=None):
    """Plot loss and/or accuracy curves for a single run.

    Returns the matplotlib Figure (creates one if ``ax`` is None).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    history = run.history
    loss_key = _loss_key(history)
    if loss_key is not None:
        ax.plot(history[loss_key], label=loss_key)
    if "accuracy" in history:
        ax.plot(history["accuracy"], label="accuracy")

    ax.set_xlabel("epoch")
    ax.set_title(run.config.get("name", run.path.name))
    ax.legend()
    return fig


def compare_runs(run_dirs: list[str | Path]):
    """Compare runs with accuracy bars and overlaid loss curves when available.

    Returns ``(bar_fig, curve_fig)``.
    """
    runs = [load_run(path) for path in run_dirs]
    names = [run.config.get("name", run.path.name) for run in runs]
    accuracies = [float(run.metrics.get("accuracy", float("nan"))) for run in runs]

    bar_fig, bar_ax = plt.subplots()
    bar_ax.bar(names, accuracies)
    bar_ax.set_ylabel("accuracy")
    bar_ax.set_title("Run accuracy comparison")
    bar_ax.tick_params(axis="x", rotation=45)
    bar_fig.tight_layout()

    curve_fig, curve_ax = plt.subplots()
    plotted_loss = False
    for run, name in zip(runs, names):
        loss_key = _loss_key(run.history)
        if loss_key is None:
            continue
        curve_ax.plot(run.history[loss_key], label=f"{name}:{loss_key}")
        plotted_loss = True
    if plotted_loss:
        curve_ax.set_xlabel("epoch")
        curve_ax.set_ylabel("loss")
        curve_ax.set_title("Loss curves")
        curve_ax.legend()
    else:
        curve_ax.set_title("No loss curves in history")
    curve_fig.tight_layout()

    return bar_fig, curve_fig


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _per_subject_curves(
    speller_dir: Path,
) -> tuple[list[int], np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Return r values and mean/std curves for char_acc and ITR from per_subject.csv."""
    per_subject_path = speller_dir / "per_subject.csv"
    if not per_subject_path.is_file():
        acc_rows = _read_csv_rows(speller_dir / "acc_vs_repeats.csv")
        rs = [int(row["r"]) for row in acc_rows]
        char_acc = np.asarray([float(row["char_acc"]) for row in acc_rows])
        itr = np.asarray([float(row["itr"]) for row in acc_rows])
        return rs, char_acc, None, itr, None

    rows = _read_csv_rows(per_subject_path)
    by_r_acc: dict[int, list[float]] = defaultdict(list)
    by_r_itr: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        r = int(row["r"])
        by_r_acc[r].append(float(row["char_acc"]))
        by_r_itr[r].append(float(row["itr"]))

    rs = sorted(by_r_acc)
    char_acc = np.asarray([np.mean(by_r_acc[r]) for r in rs])
    itr = np.asarray([np.mean(by_r_itr[r]) for r in rs])
    char_acc_std = None
    itr_std = None
    if any(len(by_r_acc[r]) > 1 for r in rs):
        char_acc_std = np.asarray([np.std(by_r_acc[r], ddof=0) for r in rs])
        itr_std = np.asarray([np.std(by_r_itr[r], ddof=0) for r in rs])
    return rs, char_acc, char_acc_std, itr, itr_std


def plot_speller_acc_vs_repeats(speller_dir: Path, ax=None):
    """Plot mean character accuracy vs repetitions from speller artifacts."""
    rs, char_acc, char_acc_std, _, _ = _per_subject_curves(speller_dir)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if char_acc_std is not None:
        ax.errorbar(rs, char_acc, yerr=char_acc_std, marker="o", capsize=3)
    else:
        ax.plot(rs, char_acc, marker="o")

    ax.set_xlabel("repetitions (r)")
    ax.set_ylabel("char accuracy")
    ax.set_title(f"speller/{speller_dir.name}")
    ax.set_xticks(rs)
    return fig


def plot_speller_itr_vs_repeats(speller_dir: Path, ax=None):
    """Plot mean ITR vs repetitions from speller artifacts."""
    rs, _, _, itr, itr_std = _per_subject_curves(speller_dir)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if itr_std is not None:
        ax.errorbar(rs, itr, yerr=itr_std, marker="o", capsize=3)
    else:
        ax.plot(rs, itr, marker="o")

    ax.set_xlabel("repetitions (r)")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title(f"speller/{speller_dir.name}")
    ax.set_xticks(rs)
    return fig


def save_speller_plots(speller_dir: Path) -> None:
    """Write acc and ITR vs repeats PNGs under ``speller_dir/plots/``."""
    plots_dir = speller_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig = plot_speller_acc_vs_repeats(speller_dir)
    fig.savefig(plots_dir / "acc_vs_repeats.png")
    plt.close(fig)

    fig = plot_speller_itr_vs_repeats(speller_dir)
    fig.savefig(plots_dir / "itr_vs_repeats.png")
    plt.close(fig)
