"""Plot helpers that read saved run artifacts only."""

from __future__ import annotations

import csv
import json
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

    early = _early_stop_summary(speller_dir)
    if early is not None and early.get("mean_repeats_used") is not None:
        mean_r = float(early["mean_repeats_used"])
        ax.axvline(
            mean_r,
            color="C1",
            linestyle="--",
            label=f"early-stop mean r={mean_r:.2f}",
        )
        ax.legend()

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


def plot_speller_per_subject_acc(speller_dir: Path, ax=None):
    """Grouped bars of per-subject character accuracy at each configured ``r``."""
    path = speller_dir / "per_subject.csv"
    if not path.is_file():
        raise FileNotFoundError(f"per_subject.csv not found under {speller_dir}")
    rows = _read_csv_rows(path)
    by_subject: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        by_subject[row["subject"] or ""][int(row["r"])] = float(row["char_acc"])
    subjects = sorted(by_subject)
    rs = sorted({r for vals in by_subject.values() for r in vals})

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x = np.arange(len(rs), dtype=float)
    width = 0.8 / max(len(subjects), 1)
    for i, subject in enumerate(subjects):
        heights = [by_subject[subject].get(r, float("nan")) for r in rs]
        offset = (i - (len(subjects) - 1) / 2) * width
        ax.bar(x + offset, heights, width=width, label=subject or "all")

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rs])
    ax.set_xlabel("repetitions (r)")
    ax.set_ylabel("char accuracy")
    ax.set_title(f"speller/{speller_dir.name} per-subject")
    if len(subjects) > 1 or (subjects and subjects[0]):
        ax.legend()
    return fig


def plot_speller_early_stop_hist(speller_dir: Path, ax=None):
    """Histogram of repeats used under the early-stop policy."""
    csv_path = speller_dir / "early_stop_repeats.csv"
    repeats: list[int] = []
    if csv_path.is_file():
        repeats = [int(row["repeats_used"]) for row in _read_csv_rows(csv_path)]
    else:
        summary = _early_stop_summary(speller_dir)
        if summary and summary.get("repeats_used"):
            repeats = [int(r) for r in summary["repeats_used"]]
    if not repeats:
        raise FileNotFoundError(
            f"early-stop repeats not found under {speller_dir} "
            "(expected early_stop_repeats.csv or speller_metrics early_stop.repeats_used)"
        )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    bins = range(min(repeats), max(repeats) + 2)
    ax.hist(repeats, bins=bins, align="left", rwidth=0.8)
    ax.set_xlabel("repeats used")
    ax.set_ylabel("count")
    ax.set_title(f"speller/{speller_dir.name} early-stop")
    return fig


def compare_speller_runs(
    speller_dirs: list[str | Path],
    *,
    allow_mixed_label_source: bool = False,
):
    """Overlay ``char_acc(r)`` curves from multiple speller artifact directories.

    Raises ``ValueError`` if ``label_source`` differs across runs unless
    ``allow_mixed_label_source`` is true.
    """
    dirs = [Path(p) for p in speller_dirs]
    if not dirs:
        raise ValueError("compare_speller_runs requires at least one speller dir")

    sources: list[str] = []
    for speller_dir in dirs:
        meta_path = speller_dir / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            sources.append(str(meta.get("label_source", "unknown")))
        else:
            sources.append("unknown")
    if not allow_mixed_label_source and len(set(sources)) > 1:
        raise ValueError(
            "compare_speller_runs refuses to overlay mixed label_source values "
            f"{sorted(set(sources))}; pass allow_mixed_label_source=true to override"
        )

    fig, ax = plt.subplots()
    for speller_dir, source in zip(dirs, sources):
        rs, char_acc, char_acc_std, _, _ = _per_subject_curves(speller_dir)
        label = f"{speller_dir.parent.parent.name}/{speller_dir.name}"
        if char_acc_std is not None:
            ax.errorbar(
                rs, char_acc, yerr=char_acc_std, marker="o", capsize=3, label=label
            )
        else:
            ax.plot(rs, char_acc, marker="o", label=label)
    ax.set_xlabel("repetitions (r)")
    ax.set_ylabel("char accuracy")
    ax.set_title("speller comparison" + (f" ({sources[0]})" if sources else ""))
    ax.legend()
    fig.tight_layout()
    return fig


def _early_stop_summary(speller_dir: Path) -> dict | None:
    metrics_path = speller_dir / "speller_metrics.json"
    if not metrics_path.is_file():
        return None
    metrics = json.loads(metrics_path.read_text())
    early = metrics.get("early_stop")
    return early if isinstance(early, dict) else None


def save_speller_plots(speller_dir: Path) -> None:
    """Write acc / ITR / per-subject / early-stop PNGs under ``speller_dir/plots/``."""
    plots_dir = speller_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig = plot_speller_acc_vs_repeats(speller_dir)
    fig.savefig(plots_dir / "acc_vs_repeats.png")
    plt.close(fig)

    fig = plot_speller_itr_vs_repeats(speller_dir)
    fig.savefig(plots_dir / "itr_vs_repeats.png")
    plt.close(fig)

    if (speller_dir / "per_subject.csv").is_file():
        fig = plot_speller_per_subject_acc(speller_dir)
        fig.savefig(plots_dir / "per_subject_acc.png")
        plt.close(fig)

    if (speller_dir / "early_stop_repeats.csv").is_file() or (
        _early_stop_summary(speller_dir) or {}
    ).get("repeats_used"):
        fig = plot_speller_early_stop_hist(speller_dir)
        fig.savefig(plots_dir / "early_stop_repeats_hist.png")
        plt.close(fig)
