"""P300 selection paradigm: grids, stimulus schedules, packing, and decode.

Shared vocabulary between the two sides that would otherwise import each
other: ``data.pipelines`` builds training packets from selections, and
``speller`` scores and evaluates them. Nothing here depends on either.

Importing this package registers the built-in protocols, so ``get_protocol``
never depends on some other module having been imported first.
"""

from pattern_recognition.selections.decode import (
    decode_from_sequence_output,
    decode_rowcol,
    decode_single_flash,
)
from pattern_recognition.selections.grids import (
    BCI3_GRID,
    CODE_TO_COL,
    CODE_TO_ROW,
    COL_CODE,
    ROW_CODE,
    SAMARA_GRID,
    GridSpec,
)
from pattern_recognition.selections.packing import (
    PROTOCOLS,
    collate_selection_packets,
    pack_selection,
)
from pattern_recognition.selections.protocols import SpellerProtocol, get_protocol
from pattern_recognition.selections.simulate import simulate_samara_selections
from pattern_recognition.selections.types import Selection

__all__ = [
    "BCI3_GRID",
    "CODE_TO_COL",
    "CODE_TO_ROW",
    "COL_CODE",
    "PROTOCOLS",
    "ROW_CODE",
    "SAMARA_GRID",
    "GridSpec",
    "Selection",
    "SpellerProtocol",
    "collate_selection_packets",
    "decode_from_sequence_output",
    "decode_rowcol",
    "decode_single_flash",
    "get_protocol",
    "pack_selection",
    "simulate_samara_selections",
]
