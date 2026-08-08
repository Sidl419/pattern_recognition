"""Plot helpers that read saved run artifacts only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

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
