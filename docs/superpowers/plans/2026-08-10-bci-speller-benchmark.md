# BCI Speller Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `pattern_recognition.speller` so a trained binary flash scorer can be evaluated as a BCI speller (`char_acc(r)`, ITR, online early-stop) for BCI3 row×col (ground truth) and Samara single-flash simulation, writing artifacts under `results/<run>/speller/<tag>/`.

**Architecture:** Protocol-agnostic `Selection` objects; `flash_scorer` scores epochs then `decode` aggregates; optional `selection_classifier` slot shares metrics only. Config validated in Pydantic (`schema.py` docstring owns the rule list). Benchmark is an evaluation stage on an existing `run_dir`, not a retrain loop.

**Tech Stack:** Python 3.10–3.11, Poetry, NumPy, PyTorch (scorer inference), Pydantic, pytest, pandas, matplotlib.

## Global Constraints

- Absolute imports only: `from pattern_recognition...`
- Spec: `docs/superpowers/specs/2026-08-09-bci-speller-benchmark-design.md`
- Do not change CNN/SVM model math or default hyperparameters
- Do not commit `Samara_data/`, `raw_data/`, large mats/CSVs
- CI tests use synthetic fixtures only (no real EEG mats required)
- Speller artifacts nest under existing binary `run_dir/speller/<tag>/`
- `label_source` always recorded (`ground_truth` | `simulated`)
- Full transformer / LLM correction are out of scope

---

## File structure (locked)

| Path | Responsibility |
|---|---|
| `pattern_recognition/speller/__init__.py` | Public exports: `run_speller_benchmark`, config types |
| `pattern_recognition/speller/types.py` | `GridSpec`, `Selection`, scorer Protocols |
| `pattern_recognition/speller/decode.py` | Row×col and single-flash decode from flash scores |
| `pattern_recognition/speller/online.py` | Cumulative decode + margin early-stop |
| `pattern_recognition/speller/simulate.py` | Samara pool → synthetic `Selection`s; stratified epoch split helper |
| `pattern_recognition/speller/metrics.py` | `char_acc`, ITR vs `r`, prediction rows, aggregates |
| `pattern_recognition/speller/schema.py` | Pydantic config + validators (docstring = rule list) |
| `pattern_recognition/speller/protocols/base.py` | Registry + `SpellerProtocol` Protocol |
| `pattern_recognition/speller/protocols/bci3_rowcol.py` | Decode + build selections from stimulus codes (real + synthetic helper) |
| `pattern_recognition/speller/protocols/samara_single_flash_sim.py` | Grid, phrase, simulate selections |
| `pattern_recognition/speller/benchmark.py` | `run_speller_benchmark` → write artifacts (+ call plots) |
| `pattern_recognition/speller/__main__.py` | CLI `python -m pattern_recognition.speller run ...` |
| `pattern_recognition/reporting/plots.py` | Add speller plot helpers |
| `pattern_recognition/reporting/load.py` | Optional `load_speller_tag(run_dir, tag)` |
| `configs/speller_bci3_within.json` | Example BCI3 config |
| `configs/speller_samara_sim_within.json` | Example Samara config |
| `tests/speller/test_*.py` | Unit + schema + smoke |

---

### Task 1: Types + decode (row×col and single-flash)

**Files:**
- Create: `pattern_recognition/speller/types.py`
- Create: `pattern_recognition/speller/decode.py`
- Create: `pattern_recognition/speller/__init__.py` (minimal)
- Test: `tests/speller/test_decode.py`

**Interfaces:**
- Produces:
  - `GridSpec(chars: tuple[str, ...], n_rows: int, n_cols: int)` with `char_at(row, col) -> str` and `index_of(char) -> int`
  - `Selection(flashes, stimulus_ids, target_char, repeat_index, meta: dict)` — arrays `np.ndarray`
  - `decode_rowcol(scores, stimulus_ids, repeat_index, r, grid: GridSpec) -> str` using codes 1–6 cols, 7–12 rows and BCI3 char map from `P300Getter` logic (duplicate map constants in `decode.py` or import from a small `pattern_recognition/speller/grids.py` — prefer `grids.py` with `BCI3_GRID` and `SAMARA_GRID`)
  - `decode_single_flash(scores, stimulus_ids, repeat_index, r, grid: GridSpec) -> str` — mean score per cell id over repeats `< r`, argmax

- [ ] **Step 1: Write failing tests**

```python
# tests/speller/test_decode.py
import numpy as np
from pattern_recognition.speller.decode import decode_rowcol, decode_single_flash
from pattern_recognition.speller.grids import BCI3_GRID, SAMARA_GRID


def test_rowcol_picks_intersection():
    # One repeat: flash codes for row of 'A' (7) and col of 'A' (1) get high scores
    stimulus_ids = np.array([7, 1, 8, 2])
    scores = np.array([10.0, 10.0, 0.0, 0.0])
    repeat_index = np.array([0, 0, 0, 0])
    assert decode_rowcol(scores, stimulus_ids, repeat_index, r=1, grid=BCI3_GRID) == "A"


def test_single_flash_argmax_cell():
    # cell index of 'J' in SAMARA_GRID gets highest mean score
    j = SAMARA_GRID.index_of("J")
    stimulus_ids = np.array([j, 0, j, 1])
    scores = np.array([1.0, 0.0, 1.0, 0.0])
    repeat_index = np.array([0, 0, 1, 1])
    assert decode_single_flash(scores, stimulus_ids, repeat_index, r=2, grid=SAMARA_GRID) == "J"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/speller/test_decode.py -v`  
Expected: FAIL (import error)

- [ ] **Step 3: Implement `grids.py`, `types.py`, `decode.py`**

```python
# pattern_recognition/speller/grids.py
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
    "A", "D", "E", "F",
    "H", "I", "J", "L",
    "N", "O", "R", "S",
    "T", "U", "W", "_",
)
SAMARA_GRID = GridSpec(SAMARA_CHARS, 4, 4)

BCI3_CHARS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_")  # row-major 6x6 as in P300Getter
BCI3_GRID = GridSpec(BCI3_CHARS, 6, 6)
# Also export ROW_CODE / COL_CODE dicts matching P300Getter
```

Implement decode with mask `repeat_index < r`.

- [ ] **Step 4: Run tests — expect PASS**

Run: `poetry run pytest tests/speller/test_decode.py -v`

- [ ] **Step 5: Commit** (only if user asked for commits; otherwise stop and note)

```bash
git add pattern_recognition/speller tests/speller/test_decode.py
git commit -m "$(cat <<'EOF'
feat(speller): add grids and flash-score decode helpers

EOF
)"
```

---

### Task 2: Online cumulative decode + early-stop

**Files:**
- Create: `pattern_recognition/speller/online.py`
- Test: `tests/speller/test_online.py`

**Interfaces:**
- Consumes: `decode_rowcol` / `decode_single_flash` (or a `decode_fn(scores, selection, r) -> str`)
- Produces:
  - `margin_from_scores(cell_or_code_scores: np.ndarray) -> float` — best − second_best
  - `online_decode(selection, scores, decode_fn, r_max, early_stop: bool, margin_tau: float | None) -> list[dict]`  
    each step: `{r, pred, margin, stopped: bool}`

- [ ] **Step 1: Failing test**

```python
# tests/speller/test_online.py
import numpy as np
from pattern_recognition.speller.online import online_decode
from pattern_recognition.speller.grids import SAMARA_GRID
from pattern_recognition.speller.types import Selection
from pattern_recognition.speller.decode import decode_single_flash

def test_early_stop_triggers_on_margin():
    j = SAMARA_GRID.index_of("J")
    # 2 cells only for simplicity: use real grid but give J huge scores from r=1
    n = 16
    stim = np.array([i for i in range(n) for _ in range(3)])
    rep = np.array([r for _ in range(n) for r in range(3)])
    scores = np.zeros(len(stim))
    scores[stim == j] = 5.0
    sel = Selection(flashes=np.zeros((len(stim), 1, 4)), stimulus_ids=stim,
                    target_char="J", repeat_index=rep, meta={"label_source": "simulated"})
    def dec(sc, s, r):
        return decode_single_flash(sc, s.stimulus_ids, s.repeat_index, r, SAMARA_GRID)
    steps = online_decode(sel, scores, dec, r_max=3, early_stop=True, margin_tau=1.0)
    assert steps[0]["stopped"] is True
    assert steps[0]["pred"] == "J"
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `online.py`**

For each `r` in `1..r_max`: call `decode_fn`; compute margin from the same aggregated per-class scores used inside decode (export a small helper from `decode.py`: `aggregate_single_flash_scores(...) -> (n_cells,)` and `aggregate_rowcol_scores(...) -> (row_scores, col_scores)` so margin is well-defined). Stop when `early_stop` and `margin >= margin_tau`.

- [ ] **Step 4: PASS**

- [ ] **Step 5: Commit** if requested

---

### Task 3: Samara simulate + stratified epoch split

**Files:**
- Create: `pattern_recognition/speller/simulate.py`
- Test: `tests/speller/test_simulate.py`

**Interfaces:**
- Produces:
  - `stratified_epoch_holdout(y: np.ndarray, epoch_holdout: float, seed: int, stratify: bool=True) -> tuple[np.ndarray, np.ndarray]` — train_idx, eval_idx
  - `simulate_samara_selections(epochs, y, eval_idx, phrase, r_max, seed, grid=SAMARA_GRID) -> list[Selection]`  
    draws without replacement within a selection from eval pools; `stimulus_ids` = cell indices; `meta["label_source"]="simulated"`; `meta["epoch_indices"]` lists source indices for leakage tests

- [ ] **Step 1: Failing tests**

```python
def test_holdout_disjoint_and_stratified():
    y = np.array([0]*70 + [1]*30)
    train, eval_ = stratified_epoch_holdout(y, 0.3, seed=0)
    assert set(train).isdisjoint(eval_)
    assert abs(y[eval_].mean() - y.mean()) < 0.15

def test_simulate_no_reuse_within_selection_and_seed_stable():
    epochs = np.random.randn(200, 1, 8)
    y = np.array([0]*160 + [1]*40)
    _, eval_idx = stratified_epoch_holdout(y, 0.5, seed=0)
    a = simulate_samara_selections(epochs, y, eval_idx, "JU", r_max=2, seed=1)
    b = simulate_samara_selections(epochs, y, eval_idx, "JU", r_max=2, seed=1)
    assert a[0].stimulus_ids.tolist() == b[0].stimulus_ids.tolist()
    # epoch indices unique within selection
    idx = a[0].meta["epoch_indices"]
    assert len(idx) == len(set(idx))
```

- [ ] **Step 2–4: Implement until PASS**

- [ ] **Step 5: Commit** if requested

---

### Task 4: Speller metrics

**Files:**
- Create: `pattern_recognition/speller/metrics.py`
- Test: `tests/speller/test_metrics.py`

**Interfaces:**
- Consumes: `compute_itr` from `pattern_recognition.training.metrics`
- Produces:
  - `selection_duration_s(n_flashes_used: int, soa_s: float) -> float`
  - `evaluate_selections(selections, scores_per_sel, decode_fn, repetitions: list[int], n_classes: int, soa_s: float, early_stop=False, margin_tau=None) -> dict` with keys:
    - `acc_vs_repeats`: list[{r, char_acc, itr}]
    - `predictions`: list rows
    - `early_stop` summary if enabled (`char_acc_early`, `mean_repeats_used`)

ITR bits/min = `compute_itr(acc, n_classes) * (60.0 / duration_s)` where duration is time for one selection at that `r` (Samara: `16 * r * soa_s`; BCI3: `12 * r * soa_s` — pass `flashes_per_repeat` into evaluator).

- [ ] **Step 1: Failing test — perfect scores ⇒ char_acc 1.0**

- [ ] **Step 2–4: Implement + PASS**

---

### Task 5: Schema validation (docstring owns rules)

**Files:**
- Create: `pattern_recognition/speller/schema.py`
- Test: `tests/speller/test_schema.py`

**Interfaces:**
- Produces: `SpellerBenchmarkConfig`, `SplitConfig`, `SimulationConfig`, `OnlineConfig`
- Class docstring of `SpellerBenchmarkConfig` must list all validation rules from the spec (copy the table in prose/bullets)

- [ ] **Step 1: Write tests matching spec**

```python
def test_samara_ok():
    SpellerBenchmarkConfig.model_validate({...samara example...})

def test_bci3_null_split_sim_ok():
    SpellerBenchmarkConfig.model_validate({...bci3 within...})

def test_bci3_rejects_epoch_holdout():
    with pytest.raises(ValidationError):
        SpellerBenchmarkConfig.model_validate({...protocol bci3, split.epoch_holdout 0.3...})

def test_bci3_rejects_simulation_phrase():
    ...

def test_samara_requires_split_and_phrase():
    ...

def test_subject_mode_inside_split_rejected():
    # use model_validator or extra=forbid on SplitConfig
    ...

def test_within_subject_rejects_test_subjects():
    ...

def test_early_stop_requires_tau():
    ...
```

- [ ] **Step 2–4: Implement validators until PASS**

Use `model_validator(mode="after")` on `SpellerBenchmarkConfig` for protocol coupling. `SplitConfig` model_config `extra="forbid"` so unknown keys like `subject_mode` fail.

---

### Task 6: Protocol registry + synthetic builders

**Files:**
- Create: `pattern_recognition/speller/protocols/__init__.py`
- Create: `pattern_recognition/speller/protocols/base.py`
- Create: `pattern_recognition/speller/protocols/bci3_rowcol.py`
- Create: `pattern_recognition/speller/protocols/samara_single_flash_sim.py`
- Test: `tests/speller/test_protocols.py`

**Interfaces:**
- Produces: `@register_protocol`, `get_protocol(name) -> SpellerProtocol`
- `SpellerProtocol`: `name`, `grid`, `label_source`, `flashes_per_repeat`, `soa_s`, `decode(scores, selection, r) -> str`, `build_synthetic_selections(**kwargs) -> list[Selection]` for tests
- Samara protocol `build_selections_from_pools(...)` wraps `simulate_samara_selections`
- BCI3 protocol `build_selection_from_codes(flashes, stimulus_ids, repeat_index, target_char, subject)` for ground-truth rows

v1 **does not** require loading full BCI3 `.mat` inside this task — synthetic coded flashes are enough for decode/benchmark smoke. Real mat wiring is Task 8.

---

### Task 7: Benchmark runner (artifacts under run_dir)

**Files:**
- Create: `pattern_recognition/speller/benchmark.py`
- Test: `tests/speller/test_benchmark_smoke.py`

**Interfaces:**
- Produces: `run_speller_benchmark(config: SpellerBenchmarkConfig | dict | path, scores_provider=...) -> Path`
- For smoke tests, pass an injectable `FlashScorer` (e.g. oracle: score = 1 on target stimulus flashes, 0 else) so no torch checkpoint needed
- Writes:
  ```text
  <run_dir>/speller/<tag>/
    config.json
    meta.json
    speller_metrics.json
    per_subject.csv
    acc_vs_repeats.csv
    predictions.csv
  ```
- Tag collision → append `_<YYYYMMDD_HHMMSS>`
- `meta.json` includes `label_source`, `subject_mode`, `protocol`, seeds, allow_* flags

- [ ] **Step 1: Failing smoke test** creating temp `run_dir` with stub `config.json` / `run_meta.json`, running benchmark with synthetic Samara selections + oracle scorer, asserting files exist and `char_acc` at high `r` is 1.0

- [ ] **Step 2–4: Implement + PASS**

Checkpoint loading from real training runs can be a function `load_flash_scorer_from_run(run_dir) -> FlashScorer` stubbed in tests; real torch load in Task 8.

---

### Task 8: Plots + reporting load helper

**Files:**
- Modify: `pattern_recognition/reporting/plots.py`
- Modify: `pattern_recognition/reporting/load.py`
- Modify: `pattern_recognition/reporting/__init__.py`
- Modify: `pattern_recognition/speller/benchmark.py` (call plots when `plots: true`)
- Test: `tests/speller/test_plots.py`

**Interfaces:**
- `plot_speller_acc_vs_repeats(speller_dir: Path, ax=None) -> Figure`
- `plot_speller_itr_vs_repeats(speller_dir: Path, ax=None) -> Figure`
- `load_speller_tag(run_dir, tag) -> dict` reading metrics JSON
- Save PNGs under `speller/<tag>/plots/`
- Use `matplotlib.use("Agg")` in tests

---

### Task 9: CLI + example configs + wire real run scorers (thin)

**Files:**
- Create: `pattern_recognition/speller/__main__.py`
- Create: `configs/speller_bci3_within.json`
- Create: `configs/speller_samara_sim_within.json`
- Modify: `pattern_recognition/speller/benchmark.py` — default path loads model from `run_dir` checkpoints using existing experiment config + `infer`-style forward to scores (logit/prob of class 1)
- Test: `tests/speller/test_cli.py` (optional argparse smoke with `--help`)

CLI:

```bash
poetry run python -m pattern_recognition.speller run \
  --run-dir results/<binary_run>/ \
  --config configs/speller_samara_sim_within.json
```

Real BCI3 selection building from `P300Getter` / mat paths: implement `BCI3RowcolProtocol.build_selections_from_run(run_dir)` **or** accept `protocol_params` with paths; if mats missing, skip with clear error (not CI-blocking).

Samara: load epochs from paths in `protocol_params` / binary config; apply shared `split`; simulate phrase.

`selection_classifier`: define Protocol + `NotImplementedError` path in benchmark when `model_mode=selection_classifier` without registered impl — keeps the slot open.

---

### Task 10: Docs touch-up

**Files:**
- Modify: `docs/superpowers/specs/2026-08-09-bci-speller-benchmark-design.md` — set Status to `Approved`
- Modify: `AGENTS.md` — one bullet that speller benchmark lives under `pattern_recognition/speller` and writes to `run_dir/speller/`
- Modify: `README.md` only if a short “Speller benchmark” subsection already fits the research entry style (keep brief)

---

## Spec coverage checklist

| Spec item | Task |
|---|---|
| `flash_scorer` decode row×col / single-flash | 1 |
| Online + early-stop margin | 2 |
| Samara simulation + holdout / leakage | 3 |
| `char_acc(r)`, ITR, predictions | 4 |
| Config validation + docstring rules | 5 |
| Protocols registry | 6 |
| Artifacts under `run_dir/speller/<tag>/` | 7 |
| Plots | 8 |
| CLI + example configs + real scorer load | 9 |
| `selection_classifier` slot | 9 (interface only) |
| subject modes within/cross/mixed | 5 (schema) + 7/9 (benchmark wiring; mixed can be thin) |
| No LLM / no heavy transformer | respected (non-goals) |

## Notes for implementers

- Prefer small pure-NumPy core; torch only at scorer boundary.
- Reuse `P300Getter` row/col dicts — do not fork conflicting maps; if importing from `data.p300` is awkward, copy constants once into `grids.py` and add a unit test that they match `P300Getter` for letter `A`.
- Do not require network or GPU in tests.
