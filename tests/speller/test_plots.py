"""Speller reporting plots, tag loading, and save_speller_plots."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pytest

from pattern_recognition.reporting import (
    compare_speller_runs,
    load_speller_tag,
    plot_speller_acc_vs_repeats,
    plot_speller_early_stop_hist,
    plot_speller_itr_vs_repeats,
    plot_speller_per_subject_acc,
    resolve_speller_tag_dir,
    save_speller_plots,
)
from pattern_recognition.speller.benchmark import OracleFlashScorer, run_speller_benchmark
from pattern_recognition.speller.protocols import get_protocol
from tests.speller.helpers import (
    samara_smoke_config,
    stub_binary_run,
    write_speller_artifacts,
)


def test_load_speller_tag_missing_raises(tmp_path):
    run_dir = tmp_path / "binary_run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_speller_tag(run_dir, "missing")


def test_resolve_speller_tag_prefers_exact(tmp_path):
    run_dir = tmp_path / "binary_run"
    exact = run_dir / "speller" / "demo"
    collision = run_dir / "speller" / "demo_20260101_000000"
    for path in (exact, collision):
        path.mkdir(parents=True)
        (path / "speller_metrics.json").write_text(
            json.dumps({"tag": path.name, "n_selections": 1, "repetitions": [1]})
            + "\n"
        )
    write_speller_artifacts(exact)
    assert resolve_speller_tag_dir(run_dir, "demo") == exact
    loaded = load_speller_tag(run_dir, "demo")
    assert loaded["n_selections"] == 2
    assert len(loaded["acc_vs_repeats"]) == 3


def test_resolve_speller_tag_falls_back_to_newest_collision(tmp_path):
    run_dir = tmp_path / "binary_run"
    older = run_dir / "speller" / "demo_20260101_000000"
    newer = run_dir / "speller" / "demo_20260202_120000"
    for path, n in ((older, 1), (newer, 2)):
        path.mkdir(parents=True)
        (path / "speller_metrics.json").write_text(
            json.dumps({"tag": path.name, "n_selections": n}) + "\n"
        )
    assert resolve_speller_tag_dir(run_dir, "demo") == newer
    assert load_speller_tag(run_dir, "demo")["n_selections"] == 2


@pytest.mark.parametrize(
    ("plot_fn", "ylabel"),
    [
        (plot_speller_acc_vs_repeats, "char accuracy"),
        (plot_speller_itr_vs_repeats, "ITR (bits/min)"),
        (plot_speller_per_subject_acc, "char accuracy"),
    ],
)
def test_speller_plot_helpers_return_figures(tmp_path, plot_fn, ylabel):
    speller_dir = tmp_path / "speller" / "demo"
    write_speller_artifacts(speller_dir)
    fig = plot_fn(speller_dir)
    assert fig is not None
    assert fig.axes[0].get_ylabel() == ylabel
    plt.close(fig)


def test_save_speller_plots_writes_expected_pngs(tmp_path):
    speller_dir = tmp_path / "speller" / "demo"
    write_speller_artifacts(speller_dir, early_stop=True)
    save_speller_plots(speller_dir)
    plots = speller_dir / "plots"
    for name in (
        "acc_vs_repeats.png",
        "itr_vs_repeats.png",
        "per_subject_acc.png",
        "early_stop_repeats_hist.png",
    ):
        assert (plots / name).is_file(), name
    # early-stop hist helper still callable on the same artifacts
    fig = plot_speller_early_stop_hist(speller_dir)
    assert fig is not None
    plt.close(fig)


def test_compare_speller_runs_overlays_curves(tmp_path):
    a = tmp_path / "run_a" / "speller" / "tag"
    b = tmp_path / "run_b" / "speller" / "tag"
    write_speller_artifacts(a, label_source="simulated")
    write_speller_artifacts(b, label_source="simulated")
    fig = compare_speller_runs([a, b])
    assert fig is not None
    assert len(fig.axes[0].lines) >= 2
    plt.close(fig)


def test_compare_speller_runs_rejects_mixed_label_source(tmp_path):
    a = tmp_path / "run_a" / "speller" / "tag"
    b = tmp_path / "run_b" / "speller" / "tag"
    write_speller_artifacts(a, label_source="simulated")
    write_speller_artifacts(b, label_source="ground_truth")
    with pytest.raises(ValueError, match="label_source"):
        compare_speller_runs([a, b])


def test_benchmark_plots_flag(tmp_path):
    run_dir = tmp_path / "binary_run"
    stub_binary_run(run_dir)
    oracle = OracleFlashScorer(get_protocol("samara_single_flash_sim").grid)

    on = run_speller_benchmark(
        samara_smoke_config(run_dir, tag="plots_on", plots=True, repetitions=[1, 2]),
        scores_provider=oracle,
    )
    assert (on / "plots" / "acc_vs_repeats.png").is_file()
    assert (on / "plots" / "itr_vs_repeats.png").is_file()

    off = run_speller_benchmark(
        samara_smoke_config(run_dir, tag="plots_off", plots=False, repetitions=[1, 2]),
        scores_provider=oracle,
    )
    assert not (off / "plots").exists()
