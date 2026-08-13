from __future__ import annotations

import numpy as np

from pattern_recognition.selections.grids import SAMARA_GRID, GridSpec
from pattern_recognition.selections.types import Selection

__all__ = ["simulate_samara_selections"]


def simulate_samara_selections(
    epochs: np.ndarray,
    y: np.ndarray,
    eval_idx: np.ndarray,
    phrase: str,
    r_max: int,
    seed: int,
    grid: GridSpec = SAMARA_GRID,
) -> list[Selection]:
    """Build synthetic Samara single-flash selections from eval epoch pools."""
    rng = np.random.default_rng(seed)
    target_pool = eval_idx[y[eval_idx] == 1]
    nontarget_pool = eval_idx[y[eval_idx] == 0]
    n_cells = grid.n_rows * grid.n_cols

    selections: list[Selection] = []
    for target_char in phrase:
        target_cell = grid.index_of(target_char)
        target_epochs = rng.choice(target_pool, size=r_max, replace=False)
        nontarget_epochs = rng.choice(
            nontarget_pool, size=(n_cells - 1) * r_max, replace=False
        )

        flashes: list[np.ndarray] = []
        stimulus_ids: list[int] = []
        repeat_index: list[int] = []
        epoch_indices: list[int] = []
        nt_offset = 0

        for cell in range(n_cells):
            if cell == target_cell:
                cell_epochs = target_epochs
            else:
                cell_epochs = nontarget_epochs[nt_offset : nt_offset + r_max]
                nt_offset += r_max

            for r, ep_idx in enumerate(cell_epochs):
                flashes.append(epochs[ep_idx])
                stimulus_ids.append(cell)
                repeat_index.append(r)
                epoch_indices.append(int(ep_idx))

        selections.append(
            Selection(
                flashes=np.stack(flashes),
                stimulus_ids=np.asarray(stimulus_ids, dtype=np.int64),
                target_char=target_char,
                repeat_index=np.asarray(repeat_index, dtype=np.int64),
                meta={
                    "label_source": "simulated",
                    "epoch_indices": epoch_indices,
                },
            )
        )

    return selections
