# BCI speller benchmark design

**Date:** 2026-08-09  
**Status:** Approved — implementation plan `docs/superpowers/plans/2026-08-10-bci-speller-benchmark.md`. Real EEG selection loading is required by default (`use_synthetic: false`); synthetic flashes are CI-only.  
**Scope:** Character-level evaluation and online-selection simulation on top of existing binary P300 models; unified benchmark API for BCI Competition III and Samara; optional slot for end-to-end selection classifiers.

**Related:** `docs/superpowers/specs/2026-08-08-package-and-experiment-runner-design.md` (binary train → `results/<run>/`).

## Goals

1. Turn binary target/non-target models into **character selection** metrics without requiring a new primary training paradigm.
2. Provide a **speller benchmark**: `character accuracy` and ITR as a function of repetition count `r`, plus an online cumulative-score simulation.
3. Support **two datasets/protocols** in one API:
   - BCI Competition III Dataset II — classic 6×6 row×column, ground-truth stimulus codes.
   - Samara P300 classic — 4×4 single-character flash; **simulated** selections when stimulus IDs are absent from `.mat`.
4. Keep evaluation **multi-subject aware**: within-subject (default), cross-subject, and mixed-pool (reserve, matching prior epoch-averaging experiments).
5. Write speller artifacts **into the trained model’s run directory** under `speller/<tag>/`.
6. Leave an explicit **extension slot** for end-to-end `selection_classifier` models (e.g. future transformer), without making them the v1 default.

## Non-goals (v1)

- Training a production transformer / heavy multiclass model (interface + optional thin baseline only).
- LLM post-correction of spelled text (Lebedev et al. bioRxiv 2025) — out of scope; may consume benchmark outputs later.
- Recovering real Samara stimulus codes or replaying exact historical online sessions from current `.mat` files.
- Changing binary model architectures or default hyperparameters.
- Merging speller training into the binary `run_experiment` loop (benchmark is an evaluation stage).
- GNN / SEED integration into the speller benchmark.

## Motivation

Classic P300 oddball ERP primarily encodes **whether a flash was attended**, not which symbol was intended. Symbol identity lives in the **stimulus schedule** (row/column codes or single-cell IDs). Therefore the primary path is:

`binary flash scorer → protocol aggregation → character`.

End-to-end “packet of flashes → symbol” models can still be useful as research baselines (Lebedev interest; possible transformers). They share the same selection objects and metrics but are a secondary `model_mode`.

Samara `.mat` files expose only binary labels (`0` non-target / `1` target), channels P3/PZ/P4, and epochs. Protocol papers/figures supply the 4×4 map and phrase `JUST_DO_IT`, but not per-epoch stimulus IDs. Character-level Samara evaluation in v1 is therefore **protocol-faithful simulation**, always tagged `label_source: simulated`.

## Architecture

```text
pattern_recognition/
  speller/
    __init__.py
    types.py              # Selection, GridSpec, SpellerPrediction
    protocols/
      base.py             # SpellerProtocol protocol/ABC + registry
      bci3_rowcol.py
      samara_single_flash_sim.py
    decode.py             # flash scores → symbol; selection logits → symbol
    simulate.py           # Samara pool → synthetic selections
    online.py             # cumulative scores, early-stop policy
    metrics.py            # char_acc(r), ITR(r), curves, confusion
    benchmark.py          # run_speller_benchmark(config) → run_dir/speller/<tag>/
    schema.py             # Pydantic config for speller stage
```

Absolute imports only (`from pattern_recognition.speller...`).

### Model modes

| Mode | Model I/O | How symbol is chosen |
|---|---|---|
| `flash_scorer` (default) | single epoch → P(target) or logit | Protocol decode (`row×col` or `argmax` over cells) |
| `selection_classifier` (slot) | full selection packet → logits over symbols | `argmax` over grid classes |

Shared across modes: the same `list[Selection]` **data** (flashes + stimulus ids + target char + repeats) and the same **metric schema** (`char_acc(r)`, ITR, predictions table). The model forward pass differs:

- `flash_scorer` scores `selection.flashes` one-by-one (or batched as independent epochs), then `decode` aggregates by protocol.
- `selection_classifier` builds a packet tensor from the whole `Selection` (stimuli × repeats × channels × time, plus optional stimulus-id encoding) and predicts the character directly.

So `Selection` is the common unit of evaluation; it is not the same tensor shape for both models.

### Contracts

```text
Selection:
  flashes: array (n_flashes, ...)     # model-ready epochs
  stimulus_ids: array (n_flashes,)    # protocol codes or cell indices
  target_char: str
  repeat_index: array (n_flashes,)    # 0..r_max-1
  meta: { subject, label_source, soa_s, grid_id, ... }

FlashScorer.predict_scores(flashes) -> (n_flashes,)
SelectionClassifier.predict_logits(batch) -> (n_sel, n_symbols)

SpellerProtocol.build_selections(...) -> list[Selection]
SpellerProtocol.decode(scores|logits, selection, r) -> predicted_char
```

## Protocols

### `bci3_rowcol`

- Grid 6×6; stimulus codes 1–6 columns, 7–12 rows (existing `P300Getter` maps).
- One selection ≈ 12 codes × up to 15 repeats (180 flashes). Evaluation at repetition `r` uses the first `r` repeats only.
- Decode: sum flash scores per row code and per column code; character = intersection of argmax row and argmax column.
- `label_source: ground_truth`.
- Subjects A/B use the competition train/test flash split; do not pool A and B in `within_subject`.

### `samara_single_flash_sim`

- Grid 4×4 (from paradigm figure):

```text
A D E F
H I J L
N O R S
T U W _
```

- Default spelled phrase: `JUST_DO_IT` (10 selections). Optional calibration character list from the paper for alternate scenarios.
- Timing for ITR: SOA 110 ms (flash duration 60 ms documented; ITR uses onset schedule).
- Each simulated selection for target cell `c` at max repeats `r_max`:
  - draw `r_max` target epochs for cell `c` from the **eval** target pool;
  - draw `r_max` non-target epochs for each of the other 15 cells from the **eval** non-target pool;
  - without replacement within a selection; fixed RNG seed for reproducibility.
- Decode: mean score per cell over the first `r` repeats → argmax over 16 cells.
- `label_source: simulated` **always** in metrics and folder metadata.
- Current Samara files lack stimulus IDs and selection boundaries; do not claim ground-truth online replay.

## Subject modes

| Mode | Behavior | Role |
|---|---|---|
| `within_subject` | Per subject: train binary (or selection) model on that subject’s train split; evaluate speller on that subject’s eval selections; aggregate mean±std across subjects | **Default**; matches real P300 calibration |
| `cross_subject` | LOSO or train on N−1 / test on holdout subject(s). BCI3: train A→test B and reverse as configured | Generalization without per-user calibration |
| `mixed_pool` | Train on pooled subjects’ train epochs; evaluate on held-out subjects and/or held-out epochs as configured | Reserve; mirrors prior epoch-averaging mixed experiments |

**Leakage rule (all modes):** epochs used to train the scorer/classifier must not appear in evaluation selections. For Samara `within_subject`, use stratified epoch holdout (default 70% train / 30% eval, configurable, fixed seed). For `mixed_pool`, document in `meta` whether holdout is by subject, by epoch, or both; default = holdout subjects for test; prefer subject holdout when possible.

**Checkpoint consistency:** evaluation selections may only use epochs outside the training set of the checkpoint in `run_dir`. Practical v1 rules:

- **BCI3:** use the competition test characters/flashes for speller eval when the binary run trained on the train `.mat` only (existing pipeline behavior).
- **Samara (preferred): shared split config.** Binary training and speller benchmark both read the same `split` block (copied into the binary `config.json` / run artifacts). The speller **recomputes** train/eval indices from that block + subject epoch order; it does not require a separate opaque index dump to be correct, though persisting resolved indices in the run dir is still recommended for audit.
- If the speller’s `split` does not match the binary run’s recorded `split`, fail with a clear error unless `allow_split_mismatch: true` (exploratory).
- `allow_train_pool_eval: true` remains an explicit escape hatch for marked exploratory runs that score on the train pool; never the default.

**Subject policy vs epoch split (one source of truth each):**

- **Top-level only:** `subject_mode`, `test_subjects` — who is train vs test across people. Never duplicate these inside `split`.
- **`split` block only:** `seed`, `epoch_holdout`, `stratify`, `val_fraction` — how epochs are partitioned *within* the train subjects’ pools (Samara). For BCI3 official mats, `split` is `null`.

| Field | Where | Meaning | Default (Samara within) | BCI3 official |
|---|---|---|---|---|
| `subject_mode` | top-level | `within_subject` \| `cross_subject` \| `mixed_pool` | `within_subject` | used |
| `test_subjects` | top-level | Holdout subject ids for cross/mixed | `null` | e.g. `["B"]` when training on A |
| `seed` | `split` | RNG for stratified epoch splits | `0` | N/A if `split` is null |
| `epoch_holdout` | `split` | Fraction of epochs held out for speller eval | `0.3` | N/A (`split: null`) |
| `stratify` | `split` | Stratify epoch split by label 0/1 | `true` | N/A |
| `val_fraction` | `split` | Fraction of *train* epochs for binary val (disjoint from holdout) | pipeline default | N/A if `split` is null |

`simulation.seed` controls packing eval epochs into cells; if omitted, fall back to `split.seed` when `split` is present, else `0`.

Always persist a **per-subject table** in artifacts, not only aggregates.

## Online simulation

For each selection, after each repeat `k = 1..r_max`:

1. Update cumulative cell/row/col scores from flashes with `repeat_index < k`.
2. Emit current predicted character.

Metrics:

- `char_acc(r)` — accuracy if stopping after exactly `r` repeats.
- `itr(r)` — Wolpaw ITR using protocol SOA / selection duration at `r`.
- `acc_vs_repeats` curve points for configured `repetitions` list (e.g. `[1, 2, 5, 10, 15]`).
- Optional early-stop (implemented in v1): stop when `best − second_best ≥ margin_tau`; report `char_acc_early`, `mean_repeats_used`. Entropy-based stop is optional nicety, not required for v1 if margin covers the use case.

Default v1: early-stop **off** for the primary `char_acc(r)` curves, but the policy **is implemented** (cheap: after each repeat compare `best − second_best` to `margin_tau`, optionally cap at `r_max`). No deferred stub — config fields drive real behavior when enabled.

## Config and CLI

Speller stage config (JSON), conceptually:

```json
{
  "tag": "samara_sim_within_r10",
  "model_mode": "flash_scorer",
  "protocol": "samara_single_flash_sim",
  "subject_mode": "within_subject",
  "test_subjects": null,
  "repetitions": [1, 2, 5, 10],
  "online": {"early_stop": false, "margin_tau": null},
  "run_dir": "results/<binary_run>/",
  "split": {
    "seed": 0,
    "epoch_holdout": 0.3,
    "stratify": true,
    "val_fraction": 0.2
  },
  "simulation": {
    "seed": 0,
    "phrase": "JUST_DO_IT"
  },
  "allow_split_mismatch": false,
  "allow_train_pool_eval": false
}
```

`split` must match the binary run’s recorded `split` (same seed and holdout params). `subject_mode` / `test_subjects` on the speller config must match the binary run’s top-level subject policy. `simulation` controls phrase and packing; it may reuse `split.seed` when `simulation.seed` is omitted.

### Config validation (Pydantic / schema checks)

Validate at load time so protocol-specific fields cannot silently mislead.

**Documentation home for these rules:** implement them as methods on the Pydantic config model(s) in `pattern_recognition/speller/schema.py` (e.g. `SpellerBenchmarkConfig` + nested `SplitConfig` / `SimulationConfig` / `OnlineConfig`), and **keep the human-readable rule list in the class docstring** (and on each `field_validator` / `model_validator` as a one-liner). The table below is design intent; the docstring on the schema class is what contributors should update when adding a rule so it does not rot only in this markdown file.

Summary of rules (mirror into `SpellerBenchmarkConfig` docstring):

| Rule | When | Behavior |
|---|---|---|
| `split` must not contain `subject_mode` / `test_subjects` | always | error if present — those fields are top-level only |
| `split` may be `null` | `protocol == bci3_rowcol` with official train/test mats | accepted; competition flash split is implied |
| `epoch_holdout` / `stratify` forbidden | `protocol == bci3_rowcol` when `split` is non-null | error if `epoch_holdout` ∈ `(0, 1]` or `stratify: true` |
| `split` required with `epoch_holdout` ∈ `(0, 1)` | `protocol == samara_single_flash_sim` and not `allow_train_pool_eval` | error if `split` missing/`null` or holdout missing/`null`/`0` |
| `simulation` block | `protocol == samara_single_flash_sim` | required: `phrase` non-empty and ⊆ grid; `seed` optional |
| `simulation` block | `protocol == bci3_rowcol` | must be omitted or `null` |
| `test_subjects` non-empty | `subject_mode == cross_subject` (non-LOSO) | error if missing/`null`/empty |
| `test_subjects` must be `null` or empty | `subject_mode == within_subject` | error if set (avoids silent ignore) |
| `online.early_stop == true` ⇒ `margin_tau` set and `> 0` | always | error otherwise |
| `model_mode == selection_classifier` | v1 | allowed; clear error if `run_dir` checkpoint is flash-only |
| `split` + top-level subject policy vs binary `run_dir` | Samara / cross default | error on mismatch unless `allow_split_mismatch` |
| `allow_train_pool_eval` | any | if true, record in `meta.json`; warn |

Example BCI3 within-subject (`split` / `simulation` null):

```json
{
  "tag": "bci3_within_r15",
  "model_mode": "flash_scorer",
  "protocol": "bci3_rowcol",
  "subject_mode": "within_subject",
  "test_subjects": null,
  "repetitions": [1, 2, 5, 10, 15],
  "online": {"early_stop": false, "margin_tau": null},
  "run_dir": "results/<binary_run>/",
  "split": null,
  "simulation": null
}
```

Example BCI3 cross-subject (still `split: null`; subject policy only at top level):

```json
{
  "tag": "bci3_A_to_B_r15",
  "model_mode": "flash_scorer",
  "protocol": "bci3_rowcol",
  "subject_mode": "cross_subject",
  "test_subjects": ["B"],
  "repetitions": [1, 2, 5, 10, 15],
  "online": {"early_stop": false, "margin_tau": null},
  "run_dir": "results/<binary_run_train_A>/",
  "split": null,
  "simulation": null
}
```

Unit tests should cover: Samara config accepted; BCI3 with `split: null` and `simulation: null` accepted; BCI3 cross with top-level `test_subjects` accepted; config with `subject_mode` inside `split` rejected; BCI3 with `epoch_holdout: 0.3` rejected; BCI3 with `simulation.phrase` rejected; Samara without `simulation.phrase` or without `split` rejected; early-stop without `margin_tau` rejected.

CLI sketch:

```bash
poetry run python -m pattern_recognition.speller run \
  --run-dir results/<binary_run>/ \
  --config configs/speller_samara_sim.json
```

`run_dir` points at an existing binary (or selection-model) experiment directory. Checkpoint and binary `metrics.json` remain unchanged.

## Artifacts layout

Speller outputs nest under the trained run:

```text
results/<name>_<timestamp>/
  config.json                 # binary / training config
  metrics.json                # binary metrics
  checkpoints/...
  speller/
    <tag>/
      config.json             # speller stage config (resolved)
      meta.json               # label_source, subject_mode, protocol, seed, device_*
      speller_metrics.json    # aggregate + summary
      per_subject.csv         # one row per subject × metric
      acc_vs_repeats.csv
      predictions.csv         # subject, selection_id, r, true, pred
      plots/                  # PNGs written by reporting helpers
        acc_vs_repeats.png
        itr_vs_repeats.png
        ...
```

Each `run_speller_benchmark` invocation creates a **new** `speller/<tag>/` (or `<tag>_<timestamp>/` if tag collides) so repeated benchmarks do not overwrite each other.

### Plots

Benchmark writes plot PNGs under `speller/<tag>/plots/` (same run dir). Implement in `pattern_recognition/reporting/` (e.g. `plot_speller_acc_vs_repeats`, `compare_speller_runs`) reading only speller artifacts — same rule as binary reporting: no recompute from raw EEG.

**v1 (always, when data present):**

| Plot | Source | Notes |
|---|---|---|
| `acc_vs_repeats.png` | `acc_vs_repeats.csv` / metrics | mean `char_acc(r)` ± std across subjects (or single curve if one subject) |
| `itr_vs_repeats.png` | same | Wolpaw ITR vs `r` |
| `per_subject_acc.png` | `per_subject.csv` | optional small-multiples or grouped bars at configured `r` values |

**v1 when early-stop enabled:**

| Plot | Source |
|---|---|
| `early_stop_repeats_hist.png` | repeats used per selection |
| annotate early-stop operating point on `acc_vs_repeats.png` if useful |

**v1 optional / if cheap:**

| Plot | Source |
|---|---|
| `confusion.png` | `predictions.csv` at a chosen `r` (default `max(repetitions)`) |

**Compare across runs:** `compare_speller_runs(run_dirs_or_speller_tags)` overlays `char_acc(r)` curves (and optionally ITR) for different models/protocols; does not mix `label_source: simulated` and `ground_truth` on one “leaderboard” plot without an explicit flag / separate panels.

Config knobs (thin): `plots: true` (default) and maybe `plots.confusion_r`; skip matplotlib write in CI unit tests unless a smoke flag is on (metrics CSVs still written).

Reporting helpers (`load_run`, tables, plots) gain optional discovery of `speller/*/speller_metrics.json` and the plot helpers above without breaking binary-only runs.

## Integration with experiment runner

1. Train binary model with existing `run_experiment` + data pipelines (`BCI3PzEpochAverage`, Samara pipelines, etc.).
2. Run `run_speller_benchmark` against that `run_dir` (metrics + plots under `speller/<tag>/`).
3. Compare models via nested speller metrics/plots (same binary hyperparams, different architectures; or same architecture, different subject modes).

`selection_classifier` is implemented for `SequenceClassifier` (selection-packet train + direct symbol decode); see `docs/superpowers/specs/2026-08-10-p300-sequence-models-design.md`.

## Testing

- Unit: decode correctness on tiny synthetic 2×2 row×col and 2×2 single-flash grids.
- Unit: Samara simulator respects without-replacement and seed stability.
- Unit: leakage check — train epoch indices ∩ eval selection epoch indices = ∅.
- Integration: smoke benchmark on synthetic selections writing `speller/<tag>/` under a temp run dir.
- Do not require real `Samara_data/` or BCI3 `.mat` in CI; use fixtures.

## Success criteria

- From a trained binary run, one config produces `char_acc(r)` for BCI3 ground-truth **or** Samara simulated protocol.
- Samara results are never silently mixed into a ground-truth leaderboard (`label_source` required).
- Within / cross / mixed subject modes are selectable; within is default.
- `selection_classifier` can plug into the same metrics path without redesign.
- Artifacts live under `results/<run>/speller/<tag>/`.

## Open questions (resolved in discussion)

| Question | Decision |
|---|---|
| Binary aggregation vs 36-class primary? | Primary = `flash_scorer` + protocol; keep `selection_classifier` slot |
| Samara without stimulus codes? | Full protocol branch via simulation; tag `simulated` |
| Train/test on Samara? | Shared `split` config (seed, epoch_holdout, stratify, …) for binary train + speller; recompute indices; optional audit dump |
| Multi-subject policy? | `within_subject` default; `cross_subject`; `mixed_pool` reserve |
| Where to write results? | Nested under trained `run_dir/speller/<tag>/` |
| LLM correction? | Out of scope v1 |

## Implementation notes (non-binding)

- Prefer extending reporting rather than duplicating metric loaders.
- Reuse `P300Getter` stimulus maps for BCI3; do not fork row/col dictionaries.
- ITR: reuse or wrap `pattern_recognition.training.metrics` Wolpaw helper with explicit selection duration from protocol SOA × flashes used.
- Keep JSON configs thin: protocol-specific details stay in protocol classes / `protocol_params`, not a fat global schema.
- Enforce protocol-aware config validation in `speller/schema.py` (see Config validation): top-level-only `subject_mode`/`test_subjects`; reject those keys inside `split`; reject BCI3+`epoch_holdout` / BCI3+`simulation`; require Samara `split`+`simulation`. **Put the full rule list in `SpellerBenchmarkConfig`’s docstring** and keep validator messages aligned with that list.
