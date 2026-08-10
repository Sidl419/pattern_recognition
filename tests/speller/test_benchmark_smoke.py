"""End-to-end speller benchmark smoke (synthetic oracle)."""

from __future__ import annotations

import csv
import json

import pytest

from pattern_recognition.speller.benchmark import OracleFlashScorer, run_speller_benchmark
from pattern_recognition.speller.protocols import get_protocol
from tests.speller.helpers import samara_smoke_config, stub_binary_run


def test_benchmark_smoke_writes_artifacts(tmp_path):
    run_dir = tmp_path / "binary_run"
    stub_binary_run(run_dir)
    oracle = OracleFlashScorer(get_protocol("samara_single_flash_sim").grid)
    cfg = samara_smoke_config(run_dir, plots=False)

    out = run_speller_benchmark(cfg, scores_provider=oracle)

    assert out == run_dir / "speller" / "smoke"
    for name in (
        "config.json",
        "meta.json",
        "speller_metrics.json",
        "per_subject.csv",
        "acc_vs_repeats.csv",
        "predictions.csv",
    ):
        assert (out / name).is_file(), name

    meta = json.loads((out / "meta.json").read_text())
    assert meta["label_source"] == "simulated"
    assert meta["subject_mode"] == "within_subject"
    assert meta["protocol"] == "samara_single_flash_sim"
    assert meta["split_seed"] == 0
    assert meta["simulation_seed"] == 0
    assert meta["allow_split_mismatch"] is False
    assert meta["allow_train_pool_eval"] is False
    assert meta["device_requested"] == "cpu"
    assert meta["device_resolved"] == "cpu"

    metrics = json.loads((out / "speller_metrics.json").read_text())
    acc_by_r = {row["r"]: row["char_acc"] for row in metrics["acc_vs_repeats"]}
    assert acc_by_r[max(cfg["repetitions"])] == pytest.approx(1.0)

    with (out / "acc_vs_repeats.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert float(rows[-1]["char_acc"]) == pytest.approx(1.0)


def test_tag_collision_appends_timestamp(tmp_path):
    run_dir = tmp_path / "binary_run"
    stub_binary_run(run_dir)
    speller_root = run_dir / "speller" / "smoke"
    speller_root.mkdir(parents=True)

    oracle = OracleFlashScorer(get_protocol("samara_single_flash_sim").grid)
    out = run_speller_benchmark(
        samara_smoke_config(run_dir, plots=False), scores_provider=oracle
    )

    assert out != speller_root
    assert out.name.startswith("smoke_")
    assert out.parent == run_dir / "speller"
