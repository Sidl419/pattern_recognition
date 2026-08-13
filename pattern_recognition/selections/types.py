from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Selection:
    flashes: np.ndarray
    stimulus_ids: np.ndarray
    target_char: str
    repeat_index: np.ndarray
    meta: dict
