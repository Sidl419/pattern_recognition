import csv
import json
from pathlib import Path

import pytest

from pattern_recognition.speller.benchmark import OracleFlashScorer, run_speller_benchmark
from pattern_recognition.speller.protocols import get_protocol


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
    (run_dir / "run_meta.json").write_text(
        json.dumps({"seed": 0, "name": "smoke_binary"}) + "\n"
    )


def _smoke_config(run_dir: Path, *, tag: str = "smoke") -> dict:
    return {
        "tag": tag,
        "model_mode": "flash_scorer",
        "protocol": "samara_single_flash_sim",
        "subject_mode": "within_subject",
        "repetitions": [1, 2, 5],
        "run_dir": str(run_dir),
        "split": {
            "seed": 0,
            "epoch_holdout": 0.3,
            "stratify": True,
            "val_fraction": 0.2,
        },
        "simulation": {"seed": 0, "phrase": "JU"},
    }


def test_benchmark_smoke_writes_artifacts(tmp_path):
    run_dir = tmp_path / "binary_run"
    _stub_binary_run(run_dir)
    proto = get_protocol("samara_single_flash_sim")
    oracle = OracleFlashScorer(proto.grid)

    out = run_speller_benchmark(
        _smoke_config(run_dir), scores_provider=oracle
    )

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

    metrics = json.loads((out / "speller_metrics.json").read_text())
    acc_by_r = {row["r"]: row["char_acc"] for row in metrics["acc_vs_repeats"]}
    assert acc_by_r[max(_smoke_config(run_dir)["repetitions"])] == pytest.approx(1.0)

    with (out / "acc_vs_repeats.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert float(rows[-1]["char_acc"]) == pytest.approx(1.0)


def test_tag_collision_appends_timestamp(tmp_path):
    run_dir = tmp_path / "binary_run"
    _stub_binary_run(run_dir)
    speller_root = run_dir / "speller" / "smoke"
    speller_root.mkdir(parents=True)

    proto = get_protocol("samara_single_flash_sim")
    oracle = OracleFlashScorer(proto.grid)
    out = run_speller_benchmark(
        _smoke_config(run_dir), scores_provider=oracle
    )

    assert out != speller_root
    assert out.name.startswith("smoke_")
    assert out.parent == run_dir / "speller"
