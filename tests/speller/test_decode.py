import numpy as np
from pattern_recognition.speller.decode import decode_rowcol, decode_single_flash
from pattern_recognition.speller.grids import BCI3_GRID, COL_CODE, ROW_CODE, SAMARA_GRID


def test_rowcol_picks_intersection():
    # One repeat: flash codes for row of 'A' (7) and col of 'A' (1) get high scores
    stimulus_ids = np.array([7, 1, 8, 2])
    scores = np.array([10.0, 10.0, 0.0, 0.0])
    repeat_index = np.array([0, 0, 0, 0])
    assert decode_rowcol(scores, stimulus_ids, repeat_index, r=1, grid=BCI3_GRID) == "A"


def test_single_flash_argmax_cell():
    # cell index of 'J' in SAMARA_GRID gets highest mean score
    j = SAMARA_GRID.index_of("J")
    stimulus_ids = np.array([j, 0, j, 1])
    scores = np.array([1.0, 0.0, 1.0, 0.0])
    repeat_index = np.array([0, 0, 1, 1])
    assert decode_single_flash(scores, stimulus_ids, repeat_index, r=2, grid=SAMARA_GRID) == "J"


def test_bci3_codes_match_p300getter_for_a():
    from mne.channels import make_standard_montage

    from pattern_recognition.data.p300 import P300Getter

    eloc = make_standard_montage("standard_1005")
    getter = P300Getter(raw_data={}, eloc=eloc, n_channels=len(eloc.ch_names))
    assert getter.row_dict["A"] == ROW_CODE["A"]
    assert getter.col_dict["A"] == COL_CODE["A"]
    assert BCI3_GRID.char_at(0, 0) == "A"
