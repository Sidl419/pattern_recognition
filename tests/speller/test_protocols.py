import numpy as np
import pytest

from pattern_recognition.speller.grids import (
    BCI3_GRID,
    COL_CODE,
    ROW_CODE,
    SAMARA_GRID,
)
from pattern_recognition.speller.protocols import get_protocol


def test_get_protocol_bci3_rowcol():
    proto = get_protocol("bci3_rowcol")
    assert proto.name == "bci3_rowcol"
    assert proto.grid is BCI3_GRID
    assert proto.label_source == "ground_truth"
    assert proto.flashes_per_repeat == 12


def test_get_protocol_samara_single_flash_sim():
    proto = get_protocol("samara_single_flash_sim")
    assert proto.name == "samara_single_flash_sim"
    assert proto.grid is SAMARA_GRID
    assert proto.label_source == "simulated"
    assert proto.flashes_per_repeat == 16


def test_unknown_protocol_raises():
    with pytest.raises(KeyError, match="Unknown protocol"):
        get_protocol("nonexistent")


def test_bci3_build_selection_from_codes():
    proto = get_protocol("bci3_rowcol")
    target = "A"
    stimulus_ids = np.array([ROW_CODE[target], COL_CODE[target]])
    repeat_index = np.array([0, 0])
    flashes = np.zeros((2, 1, 8))
    sel = proto.build_selection_from_codes(
        flashes, stimulus_ids, repeat_index, target, subject="A"
    )
    assert sel.target_char == target
    assert sel.meta["subject"] == "A"
    assert sel.meta["label_source"] == "ground_truth"


def test_bci3_decode_via_protocol():
    proto = get_protocol("bci3_rowcol")
    target = "A"
    row_code = ROW_CODE[target]
    col_code = COL_CODE[target]
    stimulus_ids = np.array([row_code, col_code, 8, 2])
    scores = np.array([10.0, 10.0, 0.0, 0.0])
    repeat_index = np.array([0, 0, 0, 0])
    flashes = np.zeros((4, 1, 8))
    sel = proto.build_selection_from_codes(
        flashes, stimulus_ids, repeat_index, target, subject="A"
    )
    assert proto.decode(scores, sel, r=1) == target


def test_bci3_build_synthetic_selections():
    proto = get_protocol("bci3_rowcol")
    selections = proto.build_synthetic_selections(
        phrase="AB", r_max=2, subject="B", seed=0
    )
    assert len(selections) == 2
    assert selections[0].target_char == "A"
    assert selections[1].target_char == "B"
    assert selections[0].meta["subject"] == "B"
    assert selections[0].meta["label_source"] == "ground_truth"
    assert len(selections[0].stimulus_ids) == proto.flashes_per_repeat * 2


def test_samara_build_selections_from_pools():
    proto = get_protocol("samara_single_flash_sim")
    epochs = np.random.randn(100, 1, 8)
    y = np.array([0] * 80 + [1] * 20)
    eval_idx = np.arange(50, 100)
    selections = proto.build_selections_from_pools(
        epochs, y, eval_idx, phrase="JU", r_max=2, seed=0
    )
    assert len(selections) == 2
    assert selections[0].target_char == "J"
    assert selections[0].meta["label_source"] == "simulated"


def test_samara_decode_via_protocol():
    from pattern_recognition.speller.types import Selection

    proto = get_protocol("samara_single_flash_sim")
    j = SAMARA_GRID.index_of("J")
    stimulus_ids = np.array([j, 0, j, 1])
    scores = np.array([1.0, 0.0, 1.0, 0.0])
    repeat_index = np.array([0, 0, 1, 1])
    flashes = np.zeros((4, 1, 8))
    sel = Selection(
        flashes=flashes,
        stimulus_ids=stimulus_ids,
        target_char="J",
        repeat_index=repeat_index,
        meta={"label_source": "simulated"},
    )
    assert proto.decode(scores, sel, r=2) == "J"


def test_samara_build_synthetic_selections():
    proto = get_protocol("samara_single_flash_sim")
    a = proto.build_synthetic_selections(phrase="JU", r_max=2, seed=1)
    b = proto.build_synthetic_selections(phrase="JU", r_max=2, seed=1)
    assert len(a) == 2
    assert a[0].stimulus_ids.tolist() == b[0].stimulus_ids.tolist()
