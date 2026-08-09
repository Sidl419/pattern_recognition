"""Samara single-flash simulated speller protocol."""

from __future__ import annotations

import numpy as np

from pattern_recognition.speller.decode import decode_single_flash
from pattern_recognition.speller.grids import SAMARA_GRID, GridSpec
from pattern_recognition.speller.protocols.base import register_protocol
from pattern_recognition.speller.simulate import (
    simulate_samara_selections,
    stratified_epoch_holdout,
)
from pattern_recognition.speller.types import Selection


@register_protocol("samara_single_flash_sim")
class SamaraSingleFlashSimProtocol:
    name = "samara_single_flash_sim"
    grid = SAMARA_GRID
    label_source = "simulated"
    flashes_per_repeat = 16
    soa_s = 0.11

    def decode(self, scores: np.ndarray, selection: Selection, r: int) -> str:
        return decode_single_flash(
            scores,
            selection.stimulus_ids,
            selection.repeat_index,
            r,
            self.grid,
        )

    def build_selections_from_pools(
        self,
        epochs: np.ndarray,
        y: np.ndarray,
        eval_idx: np.ndarray,
        phrase: str,
        r_max: int,
        seed: int,
        grid: GridSpec | None = None,
    ) -> list[Selection]:
        """Build phrase selections from eval epoch pools."""
        return simulate_samara_selections(
            epochs,
            y,
            eval_idx,
            phrase,
            r_max,
            seed,
            grid=grid or self.grid,
        )

    def build_synthetic_selections(
        self,
        *,
        phrase: str,
        r_max: int,
        seed: int = 0,
        n_epochs: int = 200,
        epoch_holdout: float = 0.3,
        flash_shape: tuple[int, ...] = (1, 8),
    ) -> list[Selection]:
        """Build simulated selections from synthetic epoch pools."""
        rng = np.random.default_rng(seed)
        epochs = rng.standard_normal((n_epochs, *flash_shape))
        n_target = max(1, n_epochs // 5)
        y = np.array([0] * (n_epochs - n_target) + [1] * n_target)
        _, eval_idx = stratified_epoch_holdout(y, epoch_holdout, seed=seed)
        return self.build_selections_from_pools(
            epochs, y, eval_idx, phrase, r_max, seed
        )
