from __future__ import annotations

import numpy as np

from pattern_recognition.speller.grids import CODE_TO_COL, CODE_TO_ROW, GridSpec

_COL_CODES = frozenset(range(1, 7))
_ROW_CODES = frozenset(range(7, 13))


def _mask_repeats(repeat_index: np.ndarray, r: int) -> np.ndarray:
    return repeat_index < r


def decode_rowcol(
    scores: np.ndarray,
    stimulus_ids: np.ndarray,
    repeat_index: np.ndarray,
    r: int,
    grid: GridSpec,
) -> str:
    mask = _mask_repeats(repeat_index, r)
    row_scores = np.zeros(grid.n_rows, dtype=float)
    col_scores = np.zeros(grid.n_cols, dtype=float)

    for score, stim_id in zip(scores[mask], stimulus_ids[mask]):
        stim_id = int(stim_id)
        if stim_id in _ROW_CODES:
            row_scores[CODE_TO_ROW[stim_id]] += score
        elif stim_id in _COL_CODES:
            col_scores[CODE_TO_COL[stim_id]] += score

    best_row = int(np.argmax(row_scores))
    best_col = int(np.argmax(col_scores))
    return grid.char_at(best_row, best_col)


def decode_single_flash(
    scores: np.ndarray,
    stimulus_ids: np.ndarray,
    repeat_index: np.ndarray,
    r: int,
    grid: GridSpec,
) -> str:
    mask = _mask_repeats(repeat_index, r)
    n_cells = grid.n_rows * grid.n_cols
    cell_scores = np.zeros(n_cells, dtype=float)
    cell_counts = np.zeros(n_cells, dtype=int)

    for score, cell_id in zip(scores[mask], stimulus_ids[mask]):
        cell_id = int(cell_id)
        cell_scores[cell_id] += score
        cell_counts[cell_id] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        cell_means = np.divide(
            cell_scores,
            cell_counts,
            out=np.zeros_like(cell_scores),
            where=cell_counts > 0,
        )

    return grid.chars[int(np.argmax(cell_means))]
