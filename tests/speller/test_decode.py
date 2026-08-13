import numpy as np
import torch
from pattern_recognition.selections.decode import (
    decode_from_sequence_output,
    decode_rowcol,
    decode_single_flash,
)
from pattern_recognition.selections.grids import (
    BCI3_GRID,
    COL_CODE,
    ROW_CODE,
    SAMARA_GRID,
)


def test_decode_rowcol_from_sequence_logits():
    row = torch.zeros(6)
    row[0] = 1.0
    col = torch.zeros(6)
    col[0] = 1.0
    out = {"row_logits": row.unsqueeze(0), "column_logits": col.unsqueeze(0)}
    assert decode_from_sequence_output(
        out, "bci3_rowcol", BCI3_GRID
    ) == BCI3_GRID.char_at(0, 0)


def test_decode_cell_from_sequence_logits():
    logits = torch.zeros(16)
    logits[3] = 1.0
    out = {"character_logits": logits.unsqueeze(0)}
    assert (
        decode_from_sequence_output(out, "samara_single_flash_sim", SAMARA_GRID)
        == SAMARA_GRID.chars[3]
    )


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
    assert (
        decode_single_flash(scores, stimulus_ids, repeat_index, r=2, grid=SAMARA_GRID)
        == "J"
    )


def test_bci3_codes_match_p300getter_for_a():
    """row/col maps on P300Getter match speller grids (no MNE FIF montage load)."""
    from types import SimpleNamespace

    from pattern_recognition.data.p300 import P300Getter

    eloc = SimpleNamespace(ch_names=[f"EEG{i:03d}" for i in range(64)])
    getter = P300Getter(raw_data={}, eloc=eloc, n_channels=64)
    assert getter.row_dict["A"] == ROW_CODE["A"]
    assert getter.col_dict["A"] == COL_CODE["A"]
    assert BCI3_GRID.char_at(0, 0) == "A"
