"""Per-subject ITR and train-pool eval behavior."""

from __future__ import annotations

import pytest

from pattern_recognition.speller.benchmark import (
    OracleFlashScorer,
    _per_subject_rows,
    run_speller_benchmark,
)
from pattern_recognition.speller.metrics import _itr_bits_per_min, selection_duration_s
from pattern_recognition.speller.protocols import get_protocol
from tests.speller.helpers import stub_binary_run


def test_per_subject_itr_uses_subject_char_acc():
    predictions = [
        {"subject": "S1", "selection_id": 0, "r": 1, "true": "A", "pred": "A"},
        {"subject": "S1", "selection_id": 1, "r": 1, "true": "B", "pred": "A"},
        {"subject": "S2", "selection_id": 2, "r": 1, "true": "A", "pred": "A"},
        {"subject": "S2", "selection_id": 3, "r": 1, "true": "B", "pred": "B"},
    ]
    rows = _per_subject_rows(
        predictions,
        repetitions=[1],
        n_classes=16,
        soa_s=0.125,
        flashes_per_repeat=16,
    )
    by_subj = {row["subject"]: row for row in rows}
    assert by_subj["S1"]["char_acc"] == pytest.approx(0.5)
    assert by_subj["S2"]["char_acc"] == pytest.approx(1.0)
    duration = selection_duration_s(16, 0.125)
    assert by_subj["S1"]["itr"] == pytest.approx(
        _itr_bits_per_min(0.5, 16, duration)
    )
    assert by_subj["S2"]["itr"] == pytest.approx(
        _itr_bits_per_min(1.0, 16, duration)
    )
    assert by_subj["S1"]["itr"] != by_subj["S2"]["itr"]


def test_allow_train_pool_eval_uses_all_epoch_indices(monkeypatch, tmp_path):
    from pattern_recognition.speller import data_loading as dl

    captured: dict = {}

    def fake_build(*args, **kwargs):
        captured["kwargs"] = kwargs
        raise RuntimeError("stop_after_kwargs")

    monkeypatch.setattr(dl, "build_samara_selections_from_dir", fake_build)

    run_dir = tmp_path / "binary_run"
    stub_binary_run(run_dir)
    data_path = tmp_path / "Samara_data"
    data_path.mkdir()

    cfg = {
        "tag": "train_pool",
        "model_mode": "flash_scorer",
        "protocol": "samara_single_flash_sim",
        "subject_mode": "within_subject",
        "repetitions": [1],
        "run_dir": str(run_dir),
        "simulation": {"seed": 0, "phrase": "JU"},
        "use_synthetic": False,
        "allow_train_pool_eval": True,
        "protocol_params": {"path": str(data_path)},
        "plots": False,
    }
    with pytest.warns(UserWarning, match="allow_train_pool_eval"):
        with pytest.raises(RuntimeError, match="stop_after_kwargs"):
            run_speller_benchmark(
                cfg,
                scores_provider=OracleFlashScorer(
                    get_protocol("samara_single_flash_sim").grid
                ),
            )

    assert captured["kwargs"]["epoch_holdout"] == 0.0
    assert captured["kwargs"].get("use_train_pool") is True
