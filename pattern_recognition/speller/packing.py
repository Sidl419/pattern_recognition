from __future__ import annotations

import numpy as np
import torch

from pattern_recognition.speller.grids import (
    BCI3_GRID,
    CODE_TO_COL,
    CODE_TO_ROW,
    COL_CODE,
    ROW_CODE,
    SAMARA_GRID,
)
from pattern_recognition.speller.types import Selection

PROTOCOLS = frozenset({"bci3_rowcol", "samara_single_flash_sim"})


def _validate_protocol(protocol: str) -> None:
    if protocol not in PROTOCOLS:
        raise ValueError(
            f"Unknown protocol {protocol!r}; expected one of {sorted(PROTOCOLS)}"
        )


def character_index(char: str, protocol: str) -> int:
    _validate_protocol(protocol)
    if protocol == "bci3_rowcol":
        return BCI3_GRID.index_of(char)
    return SAMARA_GRID.index_of(char)


def stimulus_ids_to_model_codes(ids: np.ndarray, protocol: str) -> np.ndarray:
    _validate_protocol(protocol)
    ids = np.asarray(ids, dtype=np.int64)
    if protocol == "bci3_rowcol":
        return ids
    return ids + 1


def flash_targets_for_selection(
    selection: Selection, protocol: str
) -> np.ndarray:
    _validate_protocol(protocol)
    char = selection.target_char
    if protocol == "bci3_rowcol":
        row_code = ROW_CODE[char]
        col_code = COL_CODE[char]
        targets = [
            int(stim_id == row_code or stim_id == col_code)
            for stim_id in selection.stimulus_ids
        ]
        return np.asarray(targets, dtype=np.int64)
    cell_idx = SAMARA_GRID.index_of(char)
    return (selection.stimulus_ids == cell_idx).astype(np.int64)


def _slice_selection(selection: Selection, r: int | None) -> Selection:
    if r is None:
        return selection
    keep = selection.repeat_index < r
    return Selection(
        flashes=selection.flashes[keep],
        stimulus_ids=selection.stimulus_ids[keep],
        target_char=selection.target_char,
        repeat_index=selection.repeat_index[keep],
        meta=selection.meta,
    )


def pack_selection(
    selection: Selection, *, protocol: str, r: int | None = None
) -> dict[str, torch.Tensor]:
    _validate_protocol(protocol)
    sliced = _slice_selection(selection, r)
    n_flashes = len(sliced.stimulus_ids)
    stimulus_codes = stimulus_ids_to_model_codes(sliced.stimulus_ids, protocol)
    flash_targets = flash_targets_for_selection(sliced, protocol)

    packet: dict[str, torch.Tensor] = {
        "epochs": torch.as_tensor(sliced.flashes, dtype=torch.float32),
        "stimulus_codes": torch.as_tensor(stimulus_codes, dtype=torch.long),
        "repetitions": torch.as_tensor(sliced.repeat_index, dtype=torch.long),
        "valid_mask": torch.ones(n_flashes, dtype=torch.bool),
        "flash_targets": torch.as_tensor(flash_targets, dtype=torch.float32),
        "character_target": torch.tensor(
            character_index(selection.target_char, protocol), dtype=torch.long
        ),
    }

    if protocol == "bci3_rowcol":
        char = selection.target_char
        packet["row_target"] = torch.tensor(
            CODE_TO_ROW[ROW_CODE[char]], dtype=torch.long
        )
        packet["column_target"] = torch.tensor(
            CODE_TO_COL[COL_CODE[char]], dtype=torch.long
        )

    return packet


def collate_selection_packets(items: list[dict]) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("items must be non-empty")

    batch_size = len(items)
    max_flashes = max(item["epochs"].shape[0] for item in items)
    n_channels = items[0]["epochs"].shape[1]
    n_times = items[0]["epochs"].shape[2]

    epochs = torch.zeros(
        batch_size, max_flashes, n_channels, n_times, dtype=torch.float32
    )
    stimulus_codes = torch.zeros(batch_size, max_flashes, dtype=torch.long)
    repetitions = torch.zeros(batch_size, max_flashes, dtype=torch.long)
    valid_mask = torch.zeros(batch_size, max_flashes, dtype=torch.bool)
    flash_targets = torch.zeros(batch_size, max_flashes, dtype=torch.float32)
    character_target = torch.stack([item["character_target"] for item in items])

    for i, item in enumerate(items):
        n = item["epochs"].shape[0]
        epochs[i, :n] = item["epochs"]
        stimulus_codes[i, :n] = item["stimulus_codes"]
        repetitions[i, :n] = item["repetitions"]
        valid_mask[i, :n] = item["valid_mask"]
        flash_targets[i, :n] = item["flash_targets"]

    batch: dict[str, torch.Tensor] = {
        "epochs": epochs,
        "stimulus_codes": stimulus_codes,
        "repetitions": repetitions,
        "valid_mask": valid_mask,
        "flash_targets": flash_targets,
        "character_target": character_target,
    }

    if "row_target" in items[0]:
        batch["row_target"] = torch.stack([item["row_target"] for item in items])
        batch["column_target"] = torch.stack(
            [item["column_target"] for item in items]
        )

    return batch
