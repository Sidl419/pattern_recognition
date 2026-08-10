from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from pattern_recognition.speller.decode import (
    aggregate_rowcol_scores,
    aggregate_single_flash_scores,
)
from pattern_recognition.speller.grids import GridSpec
from pattern_recognition.speller.types import Selection

DecodeMode = Literal["rowcol", "single_flash"]


def margin_from_scores(cell_or_code_scores: np.ndarray) -> float:
    if cell_or_code_scores.size == 0:
        return 0.0
    if cell_or_code_scores.size == 1:
        return float(cell_or_code_scores[0])
    top_two = np.partition(cell_or_code_scores, -2)[-2:]
    return float(np.max(top_two) - np.min(top_two))


def _margin_at_r(
    scores: np.ndarray,
    stimulus_ids: np.ndarray,
    repeat_index: np.ndarray,
    r: int,
    mode: DecodeMode,
    grid: GridSpec,
) -> float:
    if mode == "rowcol":
        row_scores, col_scores = aggregate_rowcol_scores(
            scores, stimulus_ids, repeat_index, r, grid
        )
        combined = row_scores[:, np.newaxis] + col_scores[np.newaxis, :]
        return margin_from_scores(combined.ravel())

    cell_means = aggregate_single_flash_scores(
        scores, stimulus_ids, repeat_index, r, grid
    )
    return margin_from_scores(cell_means)


def online_decode(
    selection: Selection,
    scores: np.ndarray | None,
    decode_fn: Callable[[np.ndarray, Selection, int], str],
    r_max: int,
    early_stop: bool,
    margin_tau: float | None,
    *,
    mode: DecodeMode,
    grid: GridSpec,
    score_fn: Callable[[Selection, int], np.ndarray] | None = None,
) -> list[dict]:
    """Cumulative decode for r=1..r_max with optional margin early-stop.

    ``mode`` and ``grid`` are required so margin aggregation does not guess
    row×col vs single-flash from stimulus id ranges.

    Provide either a static ``scores`` vector (independent flash scorers) or
    ``score_fn(selection, r)`` for contextual models that re-score at each ``r``.
    """
    if (scores is None) == (score_fn is None):
        raise ValueError("provide exactly one of scores or score_fn")
    steps: list[dict] = []
    for r in range(1, r_max + 1):
        step_scores = score_fn(selection, r) if score_fn is not None else scores
        assert step_scores is not None
        pred = decode_fn(step_scores, selection, r)
        margin = _margin_at_r(
            step_scores,
            selection.stimulus_ids,
            selection.repeat_index,
            r,
            mode,
            grid,
        )
        stopped = early_stop and margin_tau is not None and margin >= margin_tau
        steps.append({"r": r, "pred": pred, "margin": margin, "stopped": stopped})
        if stopped:
            break
    return steps
