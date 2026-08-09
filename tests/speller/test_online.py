import numpy as np
from pattern_recognition.speller.decode import decode_single_flash
from pattern_recognition.speller.grids import SAMARA_GRID
from pattern_recognition.speller.online import margin_from_scores, online_decode
from pattern_recognition.speller.types import Selection


def test_margin_from_scores_best_minus_second():
    assert margin_from_scores(np.array([5.0, 3.0, 1.0])) == 2.0


def test_early_stop_triggers_on_margin():
    j = SAMARA_GRID.index_of("J")
    n = 16
    stim = np.array([i for i in range(n) for _ in range(3)])
    rep = np.array([r for _ in range(n) for r in range(3)])
    scores = np.zeros(len(stim))
    scores[stim == j] = 5.0
    sel = Selection(
        flashes=np.zeros((len(stim), 1, 4)),
        stimulus_ids=stim,
        target_char="J",
        repeat_index=rep,
        meta={"label_source": "simulated"},
    )

    def dec(sc, s, r):
        return decode_single_flash(sc, s.stimulus_ids, s.repeat_index, r, SAMARA_GRID)

    steps = online_decode(sel, scores, dec, r_max=3, early_stop=True, margin_tau=1.0)
    assert steps[0]["stopped"] is True
    assert steps[0]["pred"] == "J"
    assert len(steps) == 1


def test_no_early_stop_runs_all_repeats():
    j = SAMARA_GRID.index_of("J")
    n = 16
    stim = np.array([i for i in range(n) for _ in range(2)])
    rep = np.array([r for _ in range(n) for r in range(2)])
    scores = np.zeros(len(stim))
    scores[stim == j] = 5.0
    sel = Selection(
        flashes=np.zeros((len(stim), 1, 4)),
        stimulus_ids=stim,
        target_char="J",
        repeat_index=rep,
        meta={"label_source": "simulated"},
    )

    def dec(sc, s, r):
        return decode_single_flash(sc, s.stimulus_ids, s.repeat_index, r, SAMARA_GRID)

    steps = online_decode(sel, scores, dec, r_max=2, early_stop=False, margin_tau=1.0)
    assert len(steps) == 2
    assert all(not step["stopped"] for step in steps)
