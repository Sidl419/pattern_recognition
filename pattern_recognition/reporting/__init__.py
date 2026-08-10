"""Reporting helpers that read saved experiment run directories."""

from pattern_recognition.reporting.load import (
    RunArtifacts,
    load_run,
    load_speller_tag,
    resolve_speller_tag_dir,
)
from pattern_recognition.reporting.plots import (
    compare_runs,
    compare_speller_runs,
    plot_speller_acc_vs_repeats,
    plot_speller_early_stop_hist,
    plot_speller_itr_vs_repeats,
    plot_speller_per_subject_acc,
    plot_training_curves,
    save_speller_plots,
)
from pattern_recognition.reporting.tables import metrics_table

__all__ = [
    "RunArtifacts",
    "compare_runs",
    "compare_speller_runs",
    "load_run",
    "load_speller_tag",
    "metrics_table",
    "plot_speller_acc_vs_repeats",
    "plot_speller_early_stop_hist",
    "plot_speller_itr_vs_repeats",
    "plot_speller_per_subject_acc",
    "plot_training_curves",
    "resolve_speller_tag_dir",
    "save_speller_plots",
]
