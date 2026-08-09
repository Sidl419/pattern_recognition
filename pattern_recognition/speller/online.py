from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pattern_recognition.speller.decode import (
    aggregate_rowcol_scores,
    aggregate_single_flash_scores,
    uses_rowcol_stimulus,
)
from pattern_recognition.speller.grids import CODE_TO_COL, CODE_TO_ROW, GridSpec
from pattern_recognition.speller.types import Selection


def margin_from_scores(cell_or_code_scores: np.ndarray) -> float:
    if cell_or_code_scores.size == 0:
        return 0.0
    if cell_or_code_scores.size == 1:
        return float(cell_or_code_scores[0])
    top_two = np.partition(cell_or_code_scores, -2)[-2:]
    return float(np.max(top_two) - np.min(top_two))


def _single_flash_grid(stimulus_ids: np.ndarray, repeat_index: np.ndarray, r: int) -> GridSpec:
    mask = repeat_index < r
    n_cells = int(np.max(stimulus_ids[mask])) + 1
    return GridSpec(tuple("?" * n_cells), 1, n_cells)


def _rowcol_grid() -> GridSpec:
    n_rows = max(CODE_TO_ROW.values()) + 1
    n_cols = max(CODE_TO_COL.values()) + 1
    return GridSpec(tuple("?" * (n_rows * n_cols)), n_rows, n_cols)


def _margin_at_r(
    scores: np.ndarray,
    stimulus_ids: np.ndarray,
    repeat_index: np.ndarray,
    r: int,
) -> float:
    if uses_rowcol_stimulus(stimulus_ids):
        grid = _rowcol_grid()
        row_scores, col_scores = aggregate_rowcol_scores(
            scores, stimulus_ids, repeat_index, r, grid
        )
        combined = row_scores[:, np.newaxis] + col_scores[np.newaxis, :]
        return margin_from_scores(combined.ravel())

    grid = _single_flash_grid(stimulus_ids, repeat_index, r)
    cell_means = aggregate_single_flash_scores(
        scores, stimulus_ids, repeat_index, r, grid
    )
    return margin_from_scores(cell_means)


def online_decode(
    selection: Selection,
    scores: np.ndarray,
    decode_fn: Callable[[np.ndarray, Selection, int], str],
    r_max: int,
    early_stop: bool,
    margin_tau: float | None,
) -> list[dict]:
    steps: list[dict] = []
    for r in range(1, r_max + 1):
        pred = decode_fn(scores, selection, r)
        margin = _margin_at_r(scores, selection.stimulus_ids, selection.repeat_index, r)
        stopped = early_stop and margin_tau is not None and margin >= margin_tau
        steps.append({"r": r, "pred": pred, "margin": margin, "stopped": stopped})
        if stopped:
            break
    return steps
