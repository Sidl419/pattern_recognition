import numpy as np
import torch
from pattern_recognition.speller.grids import COL_CODE, ROW_CODE
from pattern_recognition.speller.packing import (
    collate_selection_packets,
    flash_targets_for_selection,
    pack_selection,
    stimulus_ids_to_model_codes,
)
from pattern_recognition.speller.types import Selection


def test_samara_ids_shift_for_model():
    ids = np.arange(16, dtype=np.int64)
    codes = stimulus_ids_to_model_codes(ids, "samara_single_flash_sim")
    assert codes.min() == 1 and codes.max() == 16
    assert 0 not in codes


def test_bci3_ids_passthrough():
    ids = np.array([1, 7, 12], dtype=np.int64)
    codes = stimulus_ids_to_model_codes(ids, "bci3_rowcol")
    np.testing.assert_array_equal(codes, ids)


def test_flash_targets_bci3():
    char = "A"
    ids = np.array([COL_CODE[char], ROW_CODE[char], 2], dtype=np.int64)
    sel = Selection(
        flashes=np.zeros((3, 1, 8)),
        stimulus_ids=ids,
        target_char=char,
        repeat_index=np.zeros(3, dtype=np.int64),
        meta={},
    )
    t = flash_targets_for_selection(sel, "bci3_rowcol")
    np.testing.assert_array_equal(t, np.array([1, 1, 0]))


def test_pack_and_collate_pads():
    sels = []
    for n in (4, 6):
        sels.append(
            Selection(
                flashes=np.random.randn(n, 1, 8).astype(np.float32),
                stimulus_ids=np.arange(1, n + 1),
                target_char="A",
                repeat_index=np.zeros(n, dtype=np.int64),
                meta={},
            )
        )
    items = [pack_selection(s, protocol="bci3_rowcol") for s in sels]
    batch = collate_selection_packets(items)
    assert batch["epochs"].shape[0] == 2
    assert batch["epochs"].shape[1] == 6  # max S
    assert batch["valid_mask"].dtype == torch.bool
    assert batch["valid_mask"][0].sum() == 4
