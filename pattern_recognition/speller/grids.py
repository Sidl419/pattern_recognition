from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSpec:
    chars: tuple[str, ...]
    n_rows: int
    n_cols: int

    def index_of(self, char: str) -> int:
        return self.chars.index(char)

    def char_at(self, row: int, col: int) -> str:
        return self.chars[row * self.n_cols + col]


SAMARA_CHARS = (
    "A",
    "D",
    "E",
    "F",
    "H",
    "I",
    "J",
    "L",
    "N",
    "O",
    "R",
    "S",
    "T",
    "U",
    "W",
    "_",
)
SAMARA_GRID = GridSpec(SAMARA_CHARS, 4, 4)

BCI3_CHARS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_")
BCI3_GRID = GridSpec(BCI3_CHARS, 6, 6)

# Stimulus codes matching P300Getter row_dict / col_dict (char -> code).
ROW_CODE: dict[str, int] = {}
ROW_CODE.update(dict.fromkeys(["A", "B", "C", "D", "E", "F"], 7))
ROW_CODE.update(dict.fromkeys(["G", "H", "I", "J", "K", "L"], 8))
ROW_CODE.update(dict.fromkeys(["M", "N", "O", "P", "Q", "R"], 9))
ROW_CODE.update(dict.fromkeys(["S", "T", "U", "V", "W", "X"], 10))
ROW_CODE.update(dict.fromkeys(["Y", "Z", "1", "2", "3", "4"], 11))
ROW_CODE.update(dict.fromkeys(["5", "6", "7", "8", "9", "_"], 12))

COL_CODE: dict[str, int] = {}
COL_CODE.update(dict.fromkeys(["A", "G", "M", "S", "Y", "5"], 1))
COL_CODE.update(dict.fromkeys(["B", "H", "N", "T", "Z", "6"], 2))
COL_CODE.update(dict.fromkeys(["C", "I", "O", "U", "1", "7"], 3))
COL_CODE.update(dict.fromkeys(["D", "J", "P", "V", "2", "8"], 4))
COL_CODE.update(dict.fromkeys(["E", "K", "Q", "W", "3", "9"], 5))
COL_CODE.update(dict.fromkeys(["F", "L", "R", "X", "4", "_"], 6))

CODE_TO_ROW = {code: row for row, code in enumerate(range(7, 13))}
CODE_TO_COL = {code: col for col, code in enumerate(range(1, 7))}
