import numpy as np

from pattern_recognition.speller.decode import decode_single_flash
from pattern_recognition.speller.grids import SAMARA_GRID
from pattern_recognition.speller.metrics import evaluate_selections, selection_duration_s
from pattern_recognition.speller.types import Selection
from pattern_recognition.training.metrics import compute_itr


def _perfect_samara_selection(target_char: str, r_max: int) -> tuple[Selection, np.ndarray]:
    target_cell = SAMARA_GRID.index_of(target_char)
    n_cells = 16
    stim = np.array([i for i in range(n_cells) for _ in range(r_max)])
    rep = np.array([r for _ in range(n_cells) for r in range(r_max)])
    scores = np.where(stim == target_cell, 1.0, 0.0)
    sel = Selection(
        flashes=np.zeros((len(stim), 1, 4)),
        stimulus_ids=stim,
        target_char=target_char,
        repeat_index=rep,
        meta={"label_source": "simulated"},
    )
    return sel, scores


def test_selection_duration_s():
    assert selection_duration_s(16, 0.125) == 2.0


def test_perfect_scores_char_acc_one():
    sel, scores = _perfect_samara_selection("J", r_max=2)
    selections = [sel]
    scores_per_sel = [scores]

    def decode_fn(sc, selection, r):
        return decode_single_flash(
            sc, selection.stimulus_ids, selection.repeat_index, r, SAMARA_GRID
        )

    result = evaluate_selections(
        selections,
        scores_per_sel,
        decode_fn,
        repetitions=[1, 2],
        n_classes=16,
        soa_s=0.125,
        flashes_per_repeat=16,
        mode="single_flash",
        grid=SAMARA_GRID,
    )

    assert len(result["acc_vs_repeats"]) == 2
    for point in result["acc_vs_repeats"]:
        assert point["char_acc"] == 1.0
        duration = selection_duration_s(16 * point["r"], 0.125)
        expected_itr = compute_itr(1.0, 16) * (60.0 / duration)
        assert abs(point["itr"] - expected_itr) < 1e-6

    assert len(result["predictions"]) == 2
    assert all(row["true"] == "J" and row["pred"] == "J" for row in result["predictions"])
    assert "early_stop" not in result


def test_early_stop_summary_when_enabled():
    sel, scores = _perfect_samara_selection("J", r_max=3)
    selections = [sel]
    scores_per_sel = [scores]

    def decode_fn(sc, selection, r):
        return decode_single_flash(
            sc, selection.stimulus_ids, selection.repeat_index, r, SAMARA_GRID
        )

    result = evaluate_selections(
        selections,
        scores_per_sel,
        decode_fn,
        repetitions=[1, 2, 3],
        n_classes=16,
        soa_s=0.125,
        flashes_per_repeat=16,
        mode="single_flash",
        grid=SAMARA_GRID,
        early_stop=True,
        margin_tau=0.5,
    )

    assert result["early_stop"]["char_acc_early"] == 1.0
    assert result["early_stop"]["mean_repeats_used"] == 1.0
