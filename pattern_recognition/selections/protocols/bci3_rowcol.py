"""BCI Competition III row×column speller protocol."""

from __future__ import annotations

import numpy as np

from pattern_recognition.selections.decode import decode_rowcol
from pattern_recognition.selections.grids import BCI3_GRID
from pattern_recognition.selections.protocols.base import register_protocol
from pattern_recognition.selections.types import Selection

_ROW_CODES = tuple(range(7, 13))
_COL_CODES = tuple(range(1, 7))


@register_protocol("bci3_rowcol")
class BCI3RowcolProtocol:
    name = "bci3_rowcol"
    grid = BCI3_GRID
    label_source = "ground_truth"
    flashes_per_repeat = 12
    soa_s = 0.125

    def decode(self, scores: np.ndarray, selection: Selection, r: int) -> str:
        return decode_rowcol(
            scores,
            selection.stimulus_ids,
            selection.repeat_index,
            r,
            self.grid,
        )

    def build_selection_from_codes(
        self,
        flashes: np.ndarray,
        stimulus_ids: np.ndarray,
        repeat_index: np.ndarray,
        target_char: str,
        subject: str,
    ) -> Selection:
        """Wrap ground-truth coded flashes into a ``Selection``."""
        return Selection(
            flashes=np.asarray(flashes),
            stimulus_ids=np.asarray(stimulus_ids, dtype=np.int64),
            target_char=target_char,
            repeat_index=np.asarray(repeat_index, dtype=np.int64),
            meta={
                "subject": subject,
                "label_source": self.label_source,
                "soa_s": self.soa_s,
                "grid_id": self.name,
            },
        )

    def build_synthetic_selections(
        self,
        *,
        phrase: str,
        r_max: int,
        subject: str = "A",
        seed: int = 0,
        flash_shape: tuple[int, ...] = (1, 8),
    ) -> list[Selection]:
        """Build coded row×col selections for decode / benchmark smoke tests."""
        rng = np.random.default_rng(seed)
        per_repeat = list(_ROW_CODES) + list(_COL_CODES)
        selections: list[Selection] = []

        for target_char in phrase:
            flashes: list[np.ndarray] = []
            stimulus_ids: list[int] = []
            repeat_index: list[int] = []

            for r in range(r_max):
                order = rng.permutation(len(per_repeat))
                for idx in order:
                    code = per_repeat[idx]
                    flashes.append(rng.standard_normal(flash_shape))
                    stimulus_ids.append(code)
                    repeat_index.append(r)

            selections.append(
                self.build_selection_from_codes(
                    np.stack(flashes),
                    np.asarray(stimulus_ids),
                    np.asarray(repeat_index),
                    target_char,
                    subject,
                )
            )

        return selections
