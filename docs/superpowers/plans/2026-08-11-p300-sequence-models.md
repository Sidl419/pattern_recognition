# P300 Sequence Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `ContextualTransformer` and `SequenceClassifier` into train + speller so EEGNet / CT / SC can be compared on BCI3 and Samara under the existing `flash_scorer` | `selection_classifier` modes.

**Architecture:** Shared packing (`Selection` ↔ tensors); packet pipelines for train; thin task branch in `run_experiment`; CT loads as sequence-aware `FlashScorer`; SC implements the reserved `selection_classifier` provider. Spec: `docs/superpowers/specs/2026-08-10-p300-sequence-models-design.md`.

**Tech Stack:** Python, PyTorch, Pydantic, pytest, existing `pattern_recognition` registries.

## Global Constraints

- Absolute imports only: `from pattern_recognition...`
- Speller modes stay exactly `flash_scorer` | `selection_classifier` (no third mode)
- Pad code is always index `0`; BCI3 `num_stimulus_codes=13`; Samara `num_stimulus_codes=17`
- `Selection.stimulus_ids` stay protocol-native; packing maps Samara `0..15` → model `1..16`
- Do not change binary EEGNet/BaseCNN defaults or MSE train path except documented wiring fixes
- Do not commit `Samara_data/`, `raw_data/`, `processed_data/`, or large `.mat`/CSV
- CI must not require real EEG dumps (synthetic packet pipeline + `use_synthetic` speller)

---

## File structure

| File | Responsibility |
|---|---|
| `pattern_recognition/models/cnn.py` | Fix duplicate EEGNet; `num_stimulus_codes`; SC `rowcol`/`cell` heads |
| `pattern_recognition/models/__init__.py` | Register `ContextualTransformer`, `SequenceClassifier` factories |
| `pattern_recognition/speller/packing.py` | `Selection` ↔ model tensors; collate; flash/char targets |
| `pattern_recognition/data/datasets.py` | `SelectionPacketDataset` |
| `pattern_recognition/data/pipelines/synthetic_selection.py` | `SyntheticSelectionPackets` CI pipeline |
| `pattern_recognition/data/pipelines/bci3_selection.py` | `BCI3SelectionPackets` |
| `pattern_recognition/data/pipelines/samara_selection.py` | `SamaraSelectionPackets` |
| `pattern_recognition/data/pipelines/__init__.py` | Import new pipelines for registration |
| `pattern_recognition/training/sequence_loop.py` | Train/eval steps for CT and SC |
| `pattern_recognition/experiment/schema.py` | Optional thin fields if needed; prefer inferring task from model name |
| `pattern_recognition/experiment/runner.py` | Task branch; write `model_mode` in `run_meta` |
| `pattern_recognition/speller/benchmark.py` | `ContextualFlashScorer`, `RunSelectionClassifier`, load + eval branch |
| `pattern_recognition/speller/decode.py` | Direct symbol decode from SC logits |
| `configs/*.json` | Example train/speller configs |
| `README.md` | Model table + three-way comparison |
| `tests/...` | Packing, factories, runner smoke, speller providers |

---

### Task 1: Model cleanup + protocol-aware SequenceClassifier

**Files:**
- Modify: `pattern_recognition/models/cnn.py`
- Test: `tests/test_sequence_models.py`

**Interfaces:**
- Produces: `P300SequenceEncoder(..., num_stimulus_codes: int = 13)`; `SequenceClassifier(..., head_mode: Literal["rowcol","cell"] = "rowcol", n_cells: int = 16)`; keep `ContextualTransformer` API; single `EEGNet` class with `extract_features`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sequence_models.py
import torch
from pattern_recognition.models.cnn import (
    EEGNet,
    P300SequenceEncoder,
    ContextualTransformer,
    SequenceClassifier,
)


def test_eegnet_extract_features_then_forward():
    m = EEGNet(input_feat_dim=64, in_channels=1, num_classes=2)
    x = torch.randn(4, 1, 64)
    feats = m.extract_features(x)
    out = m(x)
    assert feats.ndim == 2 and out.shape == (4, 2)


def test_sequence_encoder_accepts_samara_codes():
    eeg = EEGNet(input_feat_dim=64, in_channels=1, num_classes=2)
    enc = P300SequenceEncoder(eeg, d_model=32, nhead=4, num_stimulus_codes=17, max_flashes=32)
    B, S, C, T = 2, 16, 1, 64
    epochs = torch.randn(B, S, C, T)
    codes = torch.arange(1, 17).repeat(B, 1)[:, :S]
    reps = torch.zeros(B, S, dtype=torch.long)
    emb, mask = enc(epochs, codes, reps)
    assert emb.shape == (B, S, 32) and mask.shape == (B, S)


def test_sequence_classifier_cell_head_samara():
    eeg = EEGNet(input_feat_dim=64, in_channels=1, num_classes=2)
    enc = P300SequenceEncoder(eeg, d_model=32, nhead=4, num_stimulus_codes=17, max_flashes=32)
    clf = SequenceClassifier(enc, head_mode="cell", n_cells=16)
    B, S = 2, 16
    epochs = torch.randn(B, S, 1, 64)
    codes = torch.arange(1, 17).unsqueeze(0).expand(B, -1)
    reps = torch.zeros(B, S, dtype=torch.long)
    out = clf(epochs, codes, reps)
    assert out["character_logits"].shape == (B, 16)
    assert "row_logits" not in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_sequence_models.py -v`  
Expected: FAIL (missing `num_stimulus_codes` / `head_mode` or import/shape errors)

- [ ] **Step 3: Implement**

1. Delete the first/old `EEGNet` class (~lines 68–111) so only the `extract_features` version remains.
2. Add `num_stimulus_codes: int = 13` to `P300SequenceEncoder.__init__`; use `nn.Embedding(num_stimulus_codes, ..., padding_idx=0)`; validate `0 <= codes < num_stimulus_codes`.
3. Change `SequenceClassifier` to:

```python
def __init__(
    self,
    sequence_encoder: P300SequenceEncoder,
    *,
    head_mode: str = "rowcol",  # "rowcol" | "cell"
    include_character_head: bool = True,
    n_cells: int = 16,
):
    ...
    if head_mode == "rowcol":
        self.row_classifier = nn.Linear(d_model, 6)
        self.column_classifier = nn.Linear(d_model, 6)
        self.character_classifier = (
            nn.Linear(d_model, 36) if include_character_head else None
        )
    elif head_mode == "cell":
        self.row_classifier = None
        self.column_classifier = None
        self.character_classifier = nn.Linear(d_model, n_cells)
    else:
        raise ValueError(...)
```

Update `forward` / `loss` to branch on which heads exist (Samara: CE on `character_logits` only).

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_sequence_models.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/models/cnn.py tests/test_sequence_models.py
git commit -m "feat(models): generalize sequence encoder codes and SC heads"
```

---

### Task 2: Selection packing helpers

**Files:**
- Create: `pattern_recognition/speller/packing.py`
- Test: `tests/speller/test_packing.py`

**Interfaces:**
- Consumes: `Selection`, `ROW_CODE`/`COL_CODE`/`SAMARA_GRID`
- Produces:
  - `stimulus_ids_to_model_codes(ids: np.ndarray, protocol: str) -> np.ndarray`
  - `flash_targets_for_selection(selection: Selection, protocol: str) -> np.ndarray`
  - `pack_selection(selection, *, protocol: str, r: int | None = None) -> dict[str, torch.Tensor]`
  - `collate_selection_packets(items: list[dict]) -> dict[str, torch.Tensor]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/speller/test_packing.py
import numpy as np
import torch
from pattern_recognition.speller.grids import ROW_CODE, COL_CODE, SAMARA_GRID
from pattern_recognition.speller.packing import (
    stimulus_ids_to_model_codes,
    flash_targets_for_selection,
    pack_selection,
    collate_selection_packets,
)
from pattern_recognition.speller.types import Selection


def test_samara_ids_shift_for_model():
    ids = np.arange(16, dtype=np.int64)
    codes = stimulus_ids_to_model_codes(ids, "samara_single_flash_sim")
    assert codes.min() == 1 and codes.max() == 16
    assert 0 not in codes


def test_bci3_ids_passthrough():
    ids = np.array([1, 7, 12], dtype=np.int64)
    codes = stimulus_ids_to_model_codes(ids, "bci3_rowcol")
    np.testing.assert_array_equal(codes, ids)


def test_flash_targets_bci3():
    char = "A"
    ids = np.array([COL_CODE[char], ROW_CODE[char], 2], dtype=np.int64)
    sel = Selection(
        flashes=np.zeros((3, 1, 8)),
        stimulus_ids=ids,
        target_char=char,
        repeat_index=np.zeros(3, dtype=np.int64),
        meta={},
    )
    t = flash_targets_for_selection(sel, "bci3_rowcol")
    np.testing.assert_array_equal(t, np.array([1, 1, 0]))


def test_pack_and_collate_pads():
    sels = []
    for n in (4, 6):
        sels.append(
            Selection(
                flashes=np.random.randn(n, 1, 8).astype(np.float32),
                stimulus_ids=np.arange(1, n + 1),
                target_char="A",
                repeat_index=np.zeros(n, dtype=np.int64),
                meta={},
            )
        )
    items = [pack_selection(s, protocol="bci3_rowcol") for s in sels]
    batch = collate_selection_packets(items)
    assert batch["epochs"].shape[0] == 2
    assert batch["epochs"].shape[1] == 6  # max S
    assert batch["valid_mask"].dtype == torch.bool
    assert batch["valid_mask"][0].sum() == 4
```

- [ ] **Step 2: Run tests — expect FAIL** (`ModuleNotFoundError`)

Run: `poetry run pytest tests/speller/test_packing.py -v`

- [ ] **Step 3: Implement `packing.py`**

```python
# Core behavior (full file in implementation):
# - bci3_rowcol: model codes == stimulus_ids (must be 1..12)
# - samara_single_flash_sim: model codes == stimulus_ids + 1
# - pack_selection: if r is not None, keep repeat_index < r
# - tensors: epochs, stimulus_codes, repetitions, valid_mask,
#   flash_targets, character_target; row_target/column_target for bci3
# - collate: pad S with zeros; valid_mask False on pads; stimulus_codes 0 on pads
```

Also add `character_index(char, protocol) -> int` using `BCI3_GRID` / `SAMARA_GRID`.

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/speller/packing.py tests/speller/test_packing.py
git commit -m "feat(speller): add selection packet packing helpers"
```

---

### Task 3: Register CT / SC model factories

**Files:**
- Modify: `pattern_recognition/models/__init__.py`
- Test: `tests/test_model_registry_sequence.py`

**Interfaces:**
- Consumes: factories call into `cnn.py` constructors from Task 1
- Produces: `get_model("ContextualTransformer")(**params)`, `get_model("SequenceClassifier")(**params)`
- Params convention:

```python
{
  "eegnet": {"input_feat_dim": 64, "in_channels": 1, "num_classes": 2},
  "d_model": 32,
  "nhead": 4,
  "num_layers": 2,
  "num_stimulus_codes": 13,
  "max_flashes": 180,
  "max_repetitions": 15,
  "head_mode": "rowcol",  # SC only
  "n_cells": 16,          # SC cell mode
  "include_character_head": true
}
```

- [ ] **Step 1: Failing test**

```python
import pattern_recognition.models  # noqa: F401
from pattern_recognition.models import get_model, list_models


def test_sequence_models_registered():
    assert "ContextualTransformer" in list_models()
    assert "SequenceClassifier" in list_models()
    ct = get_model("ContextualTransformer")(
        eegnet={"input_feat_dim": 64, "in_channels": 1},
        d_model=32,
        nhead=4,
        num_stimulus_codes=13,
        max_flashes=32,
    )
    sc = get_model("SequenceClassifier")(
        eegnet={"input_feat_dim": 64, "in_channels": 1},
        d_model=32,
        nhead=4,
        num_stimulus_codes=17,
        max_flashes=32,
        head_mode="cell",
        n_cells=16,
    )
    assert ct is not None and sc is not None
```

- [ ] **Step 2: Run — expect FAIL** (not in registry)

- [ ] **Step 3: Implement factories** in `__init__.py` building `EEGNet` → `P300SequenceEncoder` → CT/SC; map `n_channels` → `in_channels` via existing helper.

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/models/__init__.py tests/test_model_registry_sequence.py
git commit -m "feat(models): register ContextualTransformer and SequenceClassifier"
```

---

### Task 4: Synthetic selection packet pipeline + dataset

**Files:**
- Modify: `pattern_recognition/data/datasets.py` (add `SelectionPacketDataset`)
- Create: `pattern_recognition/data/pipelines/synthetic_selection.py`
- Modify: `pattern_recognition/data/pipelines/__init__.py`
- Test: `tests/test_synthetic_selection_pipeline.py`

**Interfaces:**
- Produces: `@register_pipeline("SyntheticSelectionPackets")` → `DatasetBundle` of dict items compatible with `collate_selection_packets`
- Params: `protocol: "bci3_rowcol"|"samara_single_flash_sim"`, `n_train`, `n_val`, `n_channels`, `n_times`, `r_max`, `seed`, phrase for samara / fixed chars for bci3

- [ ] **Step 1: Failing test**

```python
from torch.utils.data import DataLoader
from pattern_recognition.data.pipelines import get_pipeline
from pattern_recognition.speller.packing import collate_selection_packets


def test_synthetic_selection_packets_bci3():
    pipe = get_pipeline("SyntheticSelectionPackets")(
        protocol="bci3_rowcol",
        n_train=4,
        n_val=2,
        n_channels=1,
        n_times=64,
        r_max=2,
        seed=0,
    )
    bundle = pipe.build()
    loader = DataLoader(bundle.train, batch_size=2, collate_fn=collate_selection_packets)
    batch = next(iter(loader))
    assert batch["epochs"].ndim == 4
    assert batch["flash_targets"].shape[:2] == batch["epochs"].shape[:2]
    assert "row_target" in batch and "column_target" in batch
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

- `SelectionPacketDataset`: stores list of packed dicts (CPU tensors/numpy); `__getitem__` returns one dict.
- `SyntheticSelectionPackets.build`: use protocol `build_synthetic_selections` then `pack_selection` + targets for each selection; split into train/val lists.
- Register import in `pipelines/__init__.py`.

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/data/datasets.py \
  pattern_recognition/data/pipelines/synthetic_selection.py \
  pattern_recognition/data/pipelines/__init__.py \
  tests/test_synthetic_selection_pipeline.py
git commit -m "feat(data): add SyntheticSelectionPackets pipeline"
```

---

### Task 5: Sequence training loop + runner task branch

**Files:**
- Create: `pattern_recognition/training/sequence_loop.py`
- Modify: `pattern_recognition/experiment/runner.py`
- Test: `tests/test_sequence_run_artifacts.py`

**Interfaces:**
- Produces: `train_contextual_transformer(...)`, `train_sequence_classifier(...)` returning `(history, metrics_dict, time_elapsed)`
- Runner: if `model.name in {"ContextualTransformer","SequenceClassifier"}` use packet collate + sequence train; else existing binary path
- `run_meta.json` gains `"model_mode": "flash_scorer"|"selection_classifier"`
- Binary path also writes `"model_mode": "flash_scorer"` for consistency

- [ ] **Step 1: Failing integration test**

```python
import json
from pattern_recognition.experiment import run_experiment


def test_contextual_transformer_smoke_run(tmp_path):
    cfg = {
        "name": "ct_smoke",
        "seed": 0,
        "device": "cpu",
        "data": {
            "pipeline": "SyntheticSelectionPackets",
            "params": {
                "protocol": "bci3_rowcol",
                "n_train": 4,
                "n_val": 2,
                "n_channels": 1,
                "n_times": 64,
                "r_max": 2,
            },
        },
        "model": {
            "name": "ContextualTransformer",
            "params": {
                "eegnet": {"input_feat_dim": 64, "in_channels": 1},
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "num_stimulus_codes": 13,
                "max_flashes": 24,
                "max_repetitions": 15,
            },
        },
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 2,
            "num_epochs": 1,
            "step_size": 1,
            "gamma": 1.0,
            "save_model": True,
        },
        "output_dir": str(tmp_path),
    }
    run_dir = run_experiment(cfg)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["model_mode"] == "flash_scorer"
    assert (run_dir / "model.pt").is_file()


def test_sequence_classifier_smoke_run(tmp_path):
    cfg = {
        "name": "sc_smoke",
        "seed": 0,
        "device": "cpu",
        "data": {
            "pipeline": "SyntheticSelectionPackets",
            "params": {
                "protocol": "bci3_rowcol",
                "n_train": 4,
                "n_val": 2,
                "n_channels": 1,
                "n_times": 64,
                "r_max": 2,
            },
        },
        "model": {
            "name": "SequenceClassifier",
            "params": {
                "eegnet": {"input_feat_dim": 64, "in_channels": 1},
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "num_stimulus_codes": 13,
                "max_flashes": 24,
                "max_repetitions": 15,
                "head_mode": "rowcol",
                "include_character_head": True,
            },
        },
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 2,
            "num_epochs": 1,
            "step_size": 1,
            "gamma": 1.0,
            "save_model": True,
        },
        "output_dir": str(tmp_path),
    }
    run_dir = run_experiment(cfg)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["model_mode"] == "selection_classifier"
```


- [ ] **Step 2: Run — expect FAIL** (runner treats CT like binary CNN)

- [ ] **Step 3: Implement**

`sequence_loop.py`:
- CT: Adam + `ContextualTransformer.loss`; track flash accuracy on valid mask
- SC: Adam + `SequenceClassifier.loss`; track row/col or cell accuracy
- Return histories compatible with runner metric writers (reuse keys where possible; for SC map cell/char acc into `accuracy` field and document in metrics)

`runner.py`:
- Detect sequence models
- `DataLoader(..., collate_fn=collate_selection_packets)`
- Call sequence trainers
- Set `model_mode` in `run_meta`
- Validate: sequence model + non-packet pipeline → clear `ValueError`

- [ ] **Step 4: Run — expect PASS** (also `tests/test_run_artifacts.py` still PASS)

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/training/sequence_loop.py \
  pattern_recognition/experiment/runner.py \
  tests/test_sequence_run_artifacts.py
git commit -m "feat(experiment): train ContextualTransformer and SequenceClassifier"
```

---

### Task 6: Speller providers — ContextualFlashScorer + load routing

**Files:**
- Modify: `pattern_recognition/speller/benchmark.py`
- Modify: `tests/speller/test_load_flash_scorer.py`
- Create: `tests/speller/test_contextual_scorer.py`

**Interfaces:**
- Produces: `class ContextualFlashScorer` with `predict_scores(selection) -> np.ndarray`
- `load_flash_scorer_from_run`: if checkpoint model is `ContextualTransformer`, return `ContextualFlashScorer`; binary CNNs unchanged
- Still error if `model_mode == "selection_classifier"` is passed into **flash** loader (SC has its own loader in Task 7)

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from pattern_recognition.experiment import run_experiment
from pattern_recognition.speller.benchmark import load_flash_scorer_from_run
from pattern_recognition.speller.types import Selection


def _ct_run(tmp_path):
    return run_experiment({
        "name": "ct_scorer",
        "seed": 0,
        "device": "cpu",
        "data": {
            "pipeline": "SyntheticSelectionPackets",
            "params": {
                "protocol": "bci3_rowcol",
                "n_train": 4,
                "n_val": 2,
                "n_channels": 1,
                "n_times": 64,
                "r_max": 2,
            },
        },
        "model": {
            "name": "ContextualTransformer",
            "params": {
                "eegnet": {"input_feat_dim": 64, "in_channels": 1},
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "num_stimulus_codes": 13,
                "max_flashes": 24,
                "max_repetitions": 15,
            },
        },
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 2,
            "num_epochs": 1,
            "step_size": 1,
            "gamma": 1.0,
            "save_model": True,
        },
        "output_dir": str(tmp_path),
    })


def test_load_contextual_flash_scorer_from_run(tmp_path):
    run_dir = _ct_run(tmp_path)
    scorer = load_flash_scorer_from_run(run_dir, model_mode="flash_scorer")
    n = 12
    selection = Selection(
        flashes=np.random.randn(n, 1, 64).astype(np.float32),
        stimulus_ids=np.arange(1, n + 1, dtype=np.int64),
        target_char="A",
        repeat_index=np.zeros(n, dtype=np.int64),
        meta={},
    )
    scores = scorer.predict_scores(selection)
    assert scores.shape == (n,)
```

Keep `load_flash_scorer_from_run(..., model_mode="selection_classifier")` raising (SC uses `load_selection_classifier_from_run` in Task 7).

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `ContextualFlashScorer`**

```python
class ContextualFlashScorer:
    def __init__(self, model, device, protocol: str):
        self._model = model
        self._device = device
        self._protocol = protocol

    def predict_scores(self, selection: Selection) -> np.ndarray:
        packed = pack_selection(selection, protocol=self._protocol)
        # unsqueeze batch dim, run model, return logits/sigmoid scores for valid flashes
```

Infer `protocol` from speller config at benchmark time **or** from run `config.json` `data.params.protocol` (prefer run config for loader; benchmark passes protocol when constructing provider).

`load_flash_scorer_from_run`: branch on `exp_cfg.model.name`.

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/speller/benchmark.py tests/speller/test_load_flash_scorer.py \
  tests/speller/test_contextual_scorer.py
git commit -m "feat(speller): load ContextualTransformer as flash_scorer"
```

---

### Task 7: Implement `selection_classifier` slot

**Files:**
- Modify: `pattern_recognition/speller/benchmark.py`
- Modify: `pattern_recognition/speller/decode.py`
- Modify: `pattern_recognition/speller/metrics.py` (optional helper) or keep eval logic in benchmark
- Test: `tests/speller/test_selection_classifier.py`

**Interfaces:**
- Produces:
  - `decode_sequence_classifier_output(output: dict, protocol: str, grid) -> str`
  - `class RunSelectionClassifier` with `predict_char(selection: Selection, r: int) -> str`
  - `load_selection_classifier_from_run(run_dir) -> RunSelectionClassifier`
- Benchmark: when `model_mode == selection_classifier`, do **not** call `predict_scores`; loop `predict_char` per selection × `r` and write the same prediction CSV schema

- [ ] **Step 1: Failing tests**

```python
import torch
from pattern_recognition.speller.decode import decode_from_sequence_output
from pattern_recognition.speller.grids import BCI3_GRID, SAMARA_GRID


def test_decode_rowcol_from_logits():
    row = torch.zeros(6); row[0] = 1.0
    col = torch.zeros(6); col[0] = 1.0
    out = {"row_logits": row.unsqueeze(0), "column_logits": col.unsqueeze(0)}
    assert decode_from_sequence_output(out, "bci3_rowcol", BCI3_GRID) == BCI3_GRID.char_at(0, 0)


def test_decode_cell_from_logits():
    logits = torch.zeros(16); logits[3] = 1.0
    out = {"character_logits": logits.unsqueeze(0)}
    assert decode_from_sequence_output(out, "samara_single_flash_sim", SAMARA_GRID) == SAMARA_GRID.chars[3]


def test_selection_classifier_benchmark_smoke(tmp_path):
    from pattern_recognition.experiment import run_experiment
    from pattern_recognition.speller.benchmark import run_speller_benchmark

    run_dir = run_experiment({
        "name": "sc_bench",
        "seed": 0,
        "device": "cpu",
        "data": {
            "pipeline": "SyntheticSelectionPackets",
            "params": {
                "protocol": "bci3_rowcol",
                "n_train": 4,
                "n_val": 2,
                "n_channels": 1,
                "n_times": 64,
                "r_max": 2,
            },
        },
        "model": {
            "name": "SequenceClassifier",
            "params": {
                "eegnet": {"input_feat_dim": 64, "in_channels": 1},
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "num_stimulus_codes": 13,
                "max_flashes": 24,
                "max_repetitions": 15,
                "head_mode": "rowcol",
            },
        },
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 2,
            "num_epochs": 1,
            "step_size": 1,
            "gamma": 1.0,
            "save_model": True,
        },
        "output_dir": str(tmp_path),
    })
    speller_dir = run_speller_benchmark({
        "tag": "sc_syn",
        "model_mode": "selection_classifier",
        "protocol": "bci3_rowcol",
        "repetitions": [1, 2],
        "run_dir": str(run_dir),
        "use_synthetic": True,
        "plots": False,
        "protocol_params": {"phrase": "AB", "r_max": 2, "flash_shape": [1, 64]},
    })
    assert (speller_dir / "speller_metrics.json").is_file()
```


- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

`decode.py`:

```python
def decode_from_sequence_output(output: dict[str, torch.Tensor], protocol: str, grid: GridSpec) -> str:
    if protocol == "bci3_rowcol":
        # default: argmax row × argmax col (ignore character_logits unless config says otherwise — v1 default row×col)
        ...
    if protocol == "samara_single_flash_sim":
        idx = int(output["character_logits"].argmax())
        return grid.chars[idx]
```

`RunSelectionClassifier.predict_char`: `pack_selection(..., r=r)` → model → decode.

`_resolve_scorer` / `run_speller_benchmark`: branch for selection_classifier provider; evaluate without fake scores (new `_evaluate_symbol_predictor` mirroring prediction rows).

Mismatch checks:
- SC checkpoint + `flash_scorer` → error
- Binary/CT checkpoint + `selection_classifier` → error

- [ ] **Step 4: Run — expect PASS**; update/remove obsolete NotImplemented test

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/speller/benchmark.py pattern_recognition/speller/decode.py \
  tests/speller/test_selection_classifier.py tests/speller/test_load_flash_scorer.py
git commit -m "feat(speller): implement selection_classifier provider slot"
```

---

### Task 8: Real BCI3 / Samara packet pipelines + example configs + README

**Files:**
- Create: `pattern_recognition/data/pipelines/bci3_selection.py`
- Create: `pattern_recognition/data/pipelines/samara_selection.py`
- Modify: `pattern_recognition/data/pipelines/__init__.py`
- Create: `configs/bci3_contextual_transformer.json`, `configs/bci3_sequence_classifier.json`, `configs/samara_contextual_transformer.json`, `configs/samara_sequence_classifier.json` (paths as placeholders matching existing config style)
- Create: matching speller example configs with `model_mode` set appropriately
- Modify: `README.md` models section
- Test: unit tests that construct pipelines with monkeypatched/synthetic builders if mats absent; skip real-mat tests like existing speller real-data tests

**Interfaces:**
- `BCI3SelectionPackets`: load selections via existing speller data_loading / protocol; pack train chars from train mat, val from test mat (or holdout)
- `SamaraSelectionPackets`: reuse simulate + `split` holdout; metadata `label_source: simulated`

- [ ] **Step 1: Failing registry/smoke tests** (pipeline names registered; build with synthetic fallback params where possible)

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement pipelines + configs + README table** documenting the three-way comparison and pad/code conventions briefly (link to spec)

- [ ] **Step 4: `make test` / `poetry run pytest tests/ -q` — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/data/pipelines/bci3_selection.py \
  pattern_recognition/data/pipelines/samara_selection.py \
  pattern_recognition/data/pipelines/__init__.py \
  configs/ README.md tests/
git commit -m "feat: add BCI3/Samara selection pipelines and example configs"
```

---

### Task 9: Spec status + final verification

**Files:**
- Modify: `docs/superpowers/specs/2026-08-10-p300-sequence-models-design.md` (Status → Approved / implemented)
- Optional: one-line pointer in speller design “selection_classifier slot now implemented for SequenceClassifier”

- [ ] **Step 1: Run full suite**

```bash
poetry run pytest tests/ -q
```

Expected: all PASS

- [ ] **Step 2: Manually confirm comparison paths exist**

| Model | Train pipeline | Speller `model_mode` |
|---|---|---|
| EEGNet | existing binary | `flash_scorer` |
| ContextualTransformer | selection packets | `flash_scorer` |
| SequenceClassifier | selection packets | `selection_classifier` |

- [ ] **Step 3: Commit doc status**

```bash
git add docs/superpowers/specs/2026-08-10-p300-sequence-models-design.md
git commit -m "docs: mark sequence-models spec implemented"
```

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|---|---|
| Fix duplicate EEGNet; generalize codes 13/17 | 1 |
| Packing; Samara +1; pad 0 | 2 |
| Register CT/SC | 3 |
| Synthetic packets for CI | 4 |
| Train via `run_experiment` + `model_mode` in meta | 5 |
| CT as `flash_scorer` | 6 |
| SC as `selection_classifier` slot | 7 |
| BCI3 + Samara real packet pipelines; configs; README | 8 |
| Comparison table runnable | 5–8 |
| No third `model_mode` | 6–7 |
| Binary path unchanged | 5 regression |

**Placeholder scan:** cleared — Task 5–7 tests include full configs.

**Type consistency:** `pack_selection` / `collate_selection_packets` / `protocol` string literals `"bci3_rowcol"` | `"samara_single_flash_sim"` used throughout; `head_mode` `"rowcol"` | `"cell"`.
