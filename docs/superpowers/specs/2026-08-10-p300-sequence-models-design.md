# P300 sequence models — train + speller integration

**Date:** 2026-08-10  
**Status:** Approved / implemented  
**Scope:** Wire `P300SequenceEncoder`, `ContextualTransformer`, and `SequenceClassifier` (already in `pattern_recognition/models/cnn.py`) into the experiment runner and the reserved speller `selection_classifier` slot; support BCI Competition III and Samara; document architectures so the three-way comparison below is runnable.

**Related:**

- `docs/superpowers/specs/2026-08-08-package-and-experiment-runner-design.md` — binary `run_experiment` → `results/<run>/`
- `docs/superpowers/specs/2026-08-09-bci-speller-benchmark-design.md` — speller eval; `model_mode` ∈ {`flash_scorer`, `selection_classifier`}

## Goals

1. Make this comparison possible end-to-end (train + speller metrics) on **BCI3 and Samara**:

| Model | Supervision | Output | Decoder |
|---|---|---|---|
| EEGNet baseline | target / non-target per flash | binary scores (length = #flashes) | protocol accumulation |
| Contextual Transformer | target / non-target per flash | contextual binary scores (same length) | **same** accumulation |
| Sequence Classifier | row + column / character (protocol-specific) | one symbol | direct |

2. Keep the existing speller **two-mode** contract; do **not** invent a third `model_mode`:
   - EEGNet + Contextual Transformer → `flash_scorer`
   - Sequence Classifier → `selection_classifier` (implement the reserved slot)
3. Register constructible factories; train via config → `results/<name>_<timestamp>/`; evaluate via `run_speller_benchmark` against that `run_dir`.
4. Document architectures (this spec + README model table) without changing default binary EEGNet math beyond necessary wiring fixes noted below.

## Non-goals

- Changing binary `EEGNet` / `BaseCNN` training defaults (lr, MSE path, epoch-average pipelines) except documented bugfixes.
- GNN / SEED sequence training.
- LLM post-correction of spelled text.
- Recovering real Samara stimulus codes from current `.mat` files (Samara remains `label_source: simulated`).
- Hyperparameter search / multi-GPU DDP for sequence models.
- Replacing the binary epoch-averaging research track.

## Motivation

Binary flash scorers ignore within-selection context (other flashes, stimulus code, repetition index). The new modules reuse EEGNet as a flash encoder and add a Transformer over the selection packet. Contextual Transformer still emits per-flash scores so it stays comparable under the **same accumulation decoder** as EEGNet. Sequence Classifier predicts the symbol directly and fills the reserved `selection_classifier` slot from the speller design.

## Architecture overview

```text
Train (config → run_dir)
  EEGNet / BaseCNN         binary epochs     → model.pt  (flash_scorer)
  ContextualTransformer    selection packets → model.pt  (flash_scorer)
  SequenceClassifier       selection packets → model.pt  (selection_classifier)

Eval (speller on run_dir; same Selection lists)
  flash_scorer
    RunFlashScorer              independent epochs → scores → protocol decode
    ContextualFlashScorer       pack Selection → CT → scores → same decode
  selection_classifier
    RunSelectionClassifier      pack Selection → SC logits → direct symbol
```

Shared unit of evaluation remains `Selection`. Sequence models additionally need a **packed batch** for train and for contextual / selection inference.

### Model modes (unchanged enum)

| Mode | Used by | Symbol choice |
|---|---|---|
| `flash_scorer` | EEGNet, BaseCNN, ContextualTransformer | protocol accumulation (`row×col` or single-flash cell mean) |
| `selection_classifier` | SequenceClassifier | direct from model logits (protocol-specific heads) |

## Model architecture (documentation)

### EEGNet (flash encoder + baseline)

- Input: `(batch, channels, samples)`.
- Temporal conv → depthwise spatial → separable conv → flatten → linear classifier.
- `extract_features(x)` returns the vector before the classifier; `forward` = classifier on those features.
- Binary runner and `RunFlashScorer` keep using `forward` (or equivalent score extraction from class logits / probs as today).

### P300SequenceEncoder

- Reuses an `EEGNet` instance’s `extract_features` (classifier head unused).
- For each flash: project features → `d_model`, add embeddings for stimulus code, repetition index, and flash position.
- Transformer encoder over the flash sequence; optional causal mask.
- I/O: `epochs [B,S,C,T]`, `stimulus_codes [B,S]`, `repetitions [B,S]`, `valid_mask [B,S]` → contextual embeddings `[B,S,d_model]`.

**Stimulus code embedding:**

- Index `0` is **padding** (filler flash slots so batches are rectangular). Real stimuli never use `0`.
- `valid_mask` is `False` on pad slots; Transformer uses `src_key_padding_mask=~valid_mask`; CT loss masks pads.
- Table size = `num_stimulus_codes` (embedding length; pad already included as index 0):
  - BCI3: codes `0..12` → `num_stimulus_codes=13` (real `1..12`)
  - Samara: codes `0..16` → `num_stimulus_codes=17` (real cells `1..16`)
- `Selection.stimulus_ids` stay protocol-native (BCI3 `1..12`; Samara cells `0..15`). Packing maps to model codes (Samara `+1` → `1..16`); decode always uses the original `Selection.stimulus_ids`.

### ContextualTransformer

- Head on sequence embeddings → per-flash logit.
- Loss: masked `binary_cross_entropy_with_logits` on valid flashes; optional `pos_weight` estimated on train only.
- Speller: implements `FlashScorer.predict_scores(selection) -> (n_flashes,)` by packing one selection (no pad needed for S=len), running the model, returning scores aligned with `selection.flashes`.

### SequenceClassifier

- Attention-pool over valid flash embeddings → protocol heads:
  - **BCI3:** `row` (6) + `column` (6) + optional `character` (36).
  - **Samara:** cell / character head (16); **no** row×col heads (4×4 single-flash protocol).
- Loss: cross-entropy on active heads (BCI3: row+col + optional weighted char; Samara: cell/char).
- Speller: implements the reserved `selection_classifier` provider; at repetition `r`, pack only flashes with `repeat_index < r` (pad or shorter `S`), predict symbol.

### Cleanup in `cnn.py`

- Remove the dead duplicate early `EEGNet` class (the later definition already shadows it).
- Generalize hard-coded `0..12` / `Embedding(13)` checks to `num_stimulus_codes` (and matching validation).
- Do not change EEGNet convolutional defaults or BaseCNN math.

## Data: selection packet pipelines

New registered pipelines (thin JSON; protocol details in classes):

| Pipeline name | Protocol | Codes in model tensors | `label_source` |
|---|---|---|---|
| `BCI3SelectionPackets` | `bci3_rowcol` | `1..12`, pad `0` | `ground_truth` |
| `SamaraSelectionPackets` | `samara_single_flash_sim` | cells `1..16`, pad `0` | `simulated` |

Both yield the same item layout (collate pads to `max_flashes` within a batch or config cap):

```text
epochs            [S, C, T]
stimulus_codes    [S]
repetitions       [S]
valid_mask        [S]
flash_targets     [S]      # 1 if flash belongs to attended row/col (BCI3) or attended cell (Samara)
row_target        scalar   # BCI3 only (0..5); unused / omitted for Samara
column_target     scalar   # BCI3 only
character_target  scalar   # BCI3 0..35 or Samara 0..15
```

**Construction:**

- Reuse speller selection builders / data loading (`BCI3RowcolProtocol`, Samara simulate + holdout). Do not invent a second stimulus schedule.
- Samara train and speller eval share `split` (seed, `epoch_holdout`, stratify) and simulation phrase/seed rules; always record `label_source: simulated`.
- Synthetic packet smoke pipeline (or `use_synthetic` flag) for CI without real `.mat` files.

Binary epoch pipelines (`BCI3PzEpochAverage`, Samara average / time-shift, synthetic) stay as-is for EEGNet / BaseCNN.

## Training integration

**Approach:** extend `run_experiment` with a thin task branch (prefer small adapters for batch unpack + loss over a fully separate runner), same artifact layout.

| `model.name` | Data | Loss | Recorded `model_mode` |
|---|---|---|---|
| `EEGNet`, `BaseCNN`, … | existing binary pipelines | existing MSE binary path | `flash_scorer` |
| `ContextualTransformer` | selection packet pipelines | masked BCE (+ optional `pos_weight`) | `flash_scorer` |
| `SequenceClassifier` | selection packet pipelines | protocol CE heads | `selection_classifier` |

**Config:**

- `model.name` + `model.params` (encoder/`eegnet` sub-params, `d_model`, `nhead`, `num_layers`, `num_stimulus_codes`, `max_flashes`, `max_repetitions`, head flags).
- `data.pipeline` selects binary vs packet pipeline.
- Runner validates compatible pairs (e.g. CT/SC require packet pipelines; binary models require epoch datasets).
- Write `task` / `model_mode` into `run_meta.json` (and keep full `config.json`) so speller loading does not guess.

**Registry factories** in `pattern_recognition/models/__init__.py`:

- Keep `EEGNet`, `BaseCNN`.
- Add `ContextualTransformer`, `SequenceClassifier` factories that build EEGNet → `P300SequenceEncoder` → head from params.
- Optionally register `DeepConvNet` / others later; out of scope unless needed for the comparison table.

**Metrics in `metrics.json` for sequence runs:**

- Contextual: flash-level accuracy / balanced accuracy / F1 on valid flashes (plus train time, device fields).
- Sequence: row/col or cell accuracy; optional character accuracy; device fields.
- Character-level `char_acc(r)` remains the speller stage’s job (fair comparison across all three).

## Speller evaluation

Load provider from `run_dir` using recorded `model_mode` / `model.name`:

| Checkpoint | Required `model_mode` | Provider |
|---|---|---|
| Binary CNN | `flash_scorer` | existing `RunFlashScorer` |
| ContextualTransformer | `flash_scorer` | `ContextualFlashScorer` |
| SequenceClassifier | `selection_classifier` | `RunSelectionClassifier` (implement reserved slot) |

**Rules:**

- Mismatch (e.g. SC weights with `flash_scorer`) → clear error at load time.
- `selection_classifier` without a SequenceClassifier checkpoint → clear error (replace today’s generic NotImplemented where the slot is actually supported).
- Benchmark still builds `list[Selection]` via BCI3 / Samara protocols; metrics / `speller/<tag>/` artifacts unchanged.
- For CT and SC, packing helpers live in one place (e.g. `pattern_recognition/speller/packing.py` or under `data/`) used by train collate and eval providers.
- At evaluation repetition `r`, both CT and SC see only flashes with `repeat_index < r` (prefix of the selection), matching accumulation decode semantics.

**Decode for SequenceClassifier:**

- BCI3: argmax row × argmax col → char (optional: prefer `character_logits` when trained with that head and config says so — default row×col for parity with accumulation).
- Samara: argmax over 16 cell logits → char.

## Example configs (illustrative)

Train (names indicative):

- `configs/bci3_eegnet_binary.json` — existing style
- `configs/bci3_contextual_transformer.json` — `BCI3SelectionPackets` + `ContextualTransformer`
- `configs/bci3_sequence_classifier.json` — `BCI3SelectionPackets` + `SequenceClassifier`
- `configs/samara_contextual_transformer.json` / `configs/samara_sequence_classifier.json` — Samara packet pipeline + matching `num_stimulus_codes`

Speller:

- Same protocol configs as today with `model_mode: flash_scorer` for EEGNet and CT `run_dir`s.
- `model_mode: selection_classifier` for SequenceClassifier `run_dir`s.

## Package touchpoints

```text
pattern_recognition/
  models/
    cnn.py              # fix duplicate EEGNet; generalize code embedding; keep architectures
    __init__.py         # register CT + SC factories
  data/
    pipelines/
      bci3_selection.py     # BCI3SelectionPackets
      samara_selection.py   # SamaraSelectionPackets
    datasets.py             # SelectionPacketDataset + collate
  experiment/
    schema.py / runner.py   # task branch, validation, run_meta.model_mode
  training/
    loop.py or sequence_loop.py  # CT / SC train steps
  speller/
    benchmark.py        # ContextualFlashScorer; implement selection_classifier load path
    packing.py          # Selection ↔ model tensors (shared)
    decode.py           # direct symbol helpers if not already covered
configs/                # example train + speller JSONs
tests/                  # packing, factories, runner smoke, speller providers
docs/superpowers/specs/ # this file
README.md               # model table + how to run the three-way comparison
```

## Testing

- Unit: packing BCI3 / Samara `Selection` ↔ tensors (pad index 0; Samara cell shift `0..15` → `1..16`).
- Unit: factories build CT/SC; `num_stimulus_codes` validation.
- Unit: CT masked loss ignores pads; SC Samara path has no row/col requirement.
- Unit: speller provider routing — binary / CT → `flash_scorer`; SC → `selection_classifier`; mismatches error.
- Integration: synthetic packet train smoke → temp `run_dir` → speller smoke for both modes (no real `.mat` in CI).
- Regression: existing binary EEGNet / BaseCNN runner + `flash_scorer` tests still pass.

## Success criteria

1. From three trained run dirs (EEGNet, CT, SC) on the **same** protocol (BCI3 or Samara), speller configs produce comparable `char_acc(r)` / ITR artifacts.
2. CT uses accumulation decode identical in code path to EEGNet (only scores differ).
3. SC uses the reserved `selection_classifier` slot (not a new mode).
4. Samara sequence runs always tag `label_source: simulated`.
5. Architecture and comparison table are documented in this spec and reflected in README.

## Open decisions (resolved in discussion)

| Topic | Decision |
|---|---|
| Scope | Full path: register + train + speller providers + docs |
| Models | All three rows of the comparison table |
| Speller modes | Keep only `flash_scorer` and `selection_classifier` |
| CT mode | `flash_scorer` via sequence-aware FlashScorer |
| SC mode | `selection_classifier` (implement slot) |
| Datasets | BCI3 **and** Samara packet pipelines |
| Pad | Index `0` only; embedding sizes 13 (BCI3) / 17 (Samara) |
| Runner | Extend `run_experiment` with task adapters; shared artifacts |
| Samara SC heads | 16-way cell/char; no row×col |

## Implementation notes (non-binding)

- Prefer one packing module shared by train collate and speller providers.
- Record `device_requested` / `device_resolved` as today.
- Absolute imports only (`from pattern_recognition...`).
- After this spec is approved, write an implementation plan under `docs/superpowers/plans/` before coding.
