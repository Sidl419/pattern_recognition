"""Real-data selection builders (skip if mats absent)."""

from pathlib import Path

import pytest
from pattern_recognition.speller.benchmark import (
    OracleFlashScorer,
    run_speller_benchmark,
)
from pattern_recognition.speller.data_loading import (
    build_bci3_selections_from_mat,
    build_samara_selections_from_dir,
)
from pattern_recognition.speller.grids import BCI3_GRID, SAMARA_GRID

ROOT = Path(__file__).resolve().parents[2]
BCI3_TEST = ROOT / "matrix_dataset" / "Subject_A_Test.mat"
BCI3_ELOC = ROOT / "matrix_dataset" / "eloc64.loc"
SAMARA_DIR = ROOT / "Samara_data"
if not SAMARA_DIR.is_dir():
    SAMARA_DIR = ROOT / "raw_data"


@pytest.mark.skipif(
    not BCI3_TEST.is_file() or not BCI3_ELOC.is_file(), reason="BCI3 mats missing"
)
def test_bci3_real_selections_have_stimulus_codes():
    sels = build_bci3_selections_from_mat(
        BCI3_TEST,
        eloc_path=BCI3_ELOC,
        subject="A",
        apply_filter=False,
        max_repetitions=2,
        max_chars=3,
    )
    assert len(sels) == 3
    sel = sels[0]
    assert sel.flashes.ndim == 3
    assert sel.flashes.shape[1] == 1
    assert sel.flashes.shape[0] == 24  # 12 codes × 2 repeats
    assert set(sel.stimulus_ids.tolist()) == set(range(1, 13))
    assert sel.meta["label_source"] == "ground_truth"
    assert sel.target_char in BCI3_GRID.chars


@pytest.mark.skipif(
    not SAMARA_DIR.is_dir() or not any(SAMARA_DIR.glob("S*-P300_classic.mat")),
    reason="Samara mats missing",
)
def test_samara_real_pool_simulation():
    subject = "S0201" if (SAMARA_DIR / "S0201-P300_classic.mat").is_file() else None
    sels = build_samara_selections_from_dir(
        SAMARA_DIR,
        phrase="JU",
        r_max=2,
        epoch_holdout=0.3,
        seed=0,
        subjects=[subject] if subject else None,
    )
    assert len(sels) >= 2
    sel = sels[0]
    assert sel.flashes.shape[0] == 16 * 2
    assert sel.meta["label_source"] == "simulated"
    assert "subject" in sel.meta
    assert sel.target_char in SAMARA_GRID.chars


@pytest.mark.skipif(
    not BCI3_TEST.is_file() or not BCI3_ELOC.is_file(), reason="BCI3 mats missing"
)
def test_benchmark_bci3_real_with_oracle(tmp_path):
    run_dir = tmp_path / "bci3_run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        '{"name":"bci3","seed":0,"data":{"pipeline":"BCI3PzEpochAverage","params":{}}}\n'
    )
    config = {
        "tag": "bci3_real",
        "model_mode": "flash_scorer",
        "protocol": "bci3_rowcol",
        "subject_mode": "within_subject",
        "repetitions": [1, 2],
        "run_dir": str(run_dir),
        "split": None,
        "simulation": None,
        "use_synthetic": False,
        "plots": False,
        "protocol_params": {
            "test_mat": str(BCI3_TEST),
            "eloc_path": str(BCI3_ELOC),
            "subject": "A",
            "filter": False,
            "max_chars": 5,
        },
    }
    out = run_speller_benchmark(
        config, scores_provider=OracleFlashScorer(BCI3_GRID, mode="rowcol")
    )
    import json

    metrics = json.loads((out / "speller_metrics.json").read_text())
    assert metrics["acc_vs_repeats"][-1]["char_acc"] == pytest.approx(1.0)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["use_synthetic"] is False
    assert meta["label_source"] == "ground_truth"


def test_real_path_required_when_not_synthetic(tmp_path):
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text('{"name":"x","seed":0}\n')
    config = {
        "tag": "need_path",
        "model_mode": "flash_scorer",
        "protocol": "bci3_rowcol",
        "subject_mode": "within_subject",
        "repetitions": [1],
        "run_dir": str(run_dir),
        "split": None,
        "simulation": None,
        "use_synthetic": False,
        "plots": False,
    }
    with pytest.raises(FileNotFoundError, match="test_mat"):
        run_speller_benchmark(
            config, scores_provider=OracleFlashScorer(BCI3_GRID, mode="rowcol")
        )
