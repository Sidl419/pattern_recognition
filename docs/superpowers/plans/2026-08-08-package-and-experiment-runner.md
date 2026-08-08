# Package Restructure + Experiment Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn flat `src/` into an installable `pattern_recognition` package with a JSON-config experiment runner (epoch-averaging / CNN), reporting helpers, tests, example notebook, and a research README.

**Architecture:** Move existing modules into a Poetry package with absolute imports; registries map config names to data pipelines and CNN models; `run_experiment` validates config (Pydantic), resolves device, trains, and writes run artifacts; `reporting` reads those artifacts for tables/plots.

**Tech Stack:** Python 3.10–3.11, Poetry, PyTorch, MNE, scikit-learn, Pydantic, pytest, pandas, matplotlib.

## Global Constraints

- Absolute package imports only: `from pattern_recognition...` (never relative-dot, never flat `from utils import ...`).
- Do not change model math or default hyperparameters unless fixing a clear bug (document any intentional fix).
- Device: `"auto" | "cpu" | "cuda" | "cuda:N"`; explicit CUDA hard-fails if unavailable; record `device_requested` + `device_resolved`.
- Notebooks: prefer new example notebook; optional cleanup of old notebooks; no requirement to migrate all historical notebooks.
- Delete flat `src/` after the package works (no long-lived shims).
- Add `pydantic` dependency and `pytest` as a dev dependency.
- Spec: `docs/superpowers/specs/2026-08-08-package-and-experiment-runner-design.md`.

---

## File structure (locked)

| Path | Responsibility |
|---|---|
| `pattern_recognition/__init__.py` | Package version / public re-exports (minimal) |
| `pattern_recognition/data/p300.py` | `P300Getter` (+ motor helpers if still needed) from `src/utils.py` |
| `pattern_recognition/data/datasets.py` | `CNNMatrixDataset`, `GraphMatrixDataset`, `EEGDataset` from `src/data.py` |
| `pattern_recognition/data/time_shift.py` | Move `src/time_shift.py` as-is (absolute imports only if any) |
| `pattern_recognition/data/pipelines/base.py` | `DatasetBundle` dataclass + `DataPipeline` Protocol |
| `pattern_recognition/data/pipelines/registry.py` | `@register_pipeline`, `get_pipeline` |
| `pattern_recognition/data/pipelines/samara_average.py` | `SamaraWithinSubjectAverage` |
| `pattern_recognition/data/pipelines/samara_time_shift.py` | `SamaraTimeShift` |
| `pattern_recognition/data/pipelines/bci3_pz.py` | `BCI3PzEpochAverage` |
| `pattern_recognition/data/pipelines/synthetic.py` | Tiny synthetic pipeline for smoke tests |
| `pattern_recognition/models/cnn.py` | Move `src/models_cnn.py` |
| `pattern_recognition/models/gnn.py` | Move `src/models_gnn.py` |
| `pattern_recognition/models/registry.py` | `@register_model`, `get_model` (CNN v1) |
| `pattern_recognition/graph/electrodes.py` | Move `src/graph.py` |
| `pattern_recognition/training/metrics.py` | `compute_itr` + shared metric helpers |
| `pattern_recognition/training/loop.py` | `train_model`, `validate_model`, `infer_model`, plot helpers from utils |
| `pattern_recognition/training/device.py` | `resolve_device(requested: str) -> tuple[str, torch.device]` |
| `pattern_recognition/losses.py` | Move `src/losses.py` |
| `pattern_recognition/interpretation.py` | Move `src/interpretation.py` |
| `pattern_recognition/experiment/schema.py` | Pydantic config models |
| `pattern_recognition/experiment/runner.py` | `run_experiment` + artifact writer |
| `pattern_recognition/experiment/__main__.py` | CLI |
| `pattern_recognition/reporting/load.py` | `RunArtifacts`, `load_run` |
| `pattern_recognition/reporting/tables.py` | `metrics_table` |
| `pattern_recognition/reporting/plots.py` | `plot_training_curves`, `compare_runs` |
| `configs/*.json` | Example experiment configs |
| `tests/` | Format, registry, unit, device, smoke |
| `notebooks/examples/run_experiment_samara.ipynb` | Example + Colab + comparison |
| `README.md` | Research entry point |
| Delete: `src/` | After package green |

---

### Task 1: Package skeleton + move core modules + deps

**Files:**
- Create: `pattern_recognition/` tree as above for moved modules (`__init__.py` files empty or minimal)
- Modify: `pyproject.toml` (add pydantic; pytest under `[tool.poetry.group.dev.dependencies]`)
- Create: `tests/test_imports.py`
- Delete (end of task or Task 8): leave `src/` until Task 8

**Interfaces:**
- Produces: importable package `pattern_recognition` with modules:
  - `pattern_recognition.data.p300.P300Getter`
  - `pattern_recognition.data.datasets.{CNNMatrixDataset,GraphMatrixDataset,EEGDataset}`
  - `pattern_recognition.data.time_shift` (all public functions from current `src/time_shift.py`)
  - `pattern_recognition.models.cnn` / `pattern_recognition.models.gnn` (all classes preserved)
  - `pattern_recognition.graph.electrodes` (graph helpers)
  - `pattern_recognition.training.metrics.compute_itr`
  - `pattern_recognition.training.loop.{train_model,validate_model,infer_model,plot_sample,show_progress,...}`
  - `pattern_recognition.losses.GraphLoss`
  - `pattern_recognition.interpretation` (unchanged API)

- [ ] **Step 1: Write the failing import test**

```python
# tests/test_imports.py
def test_package_imports():
    from pattern_recognition.training.metrics import compute_itr
    from pattern_recognition.models.cnn import BaseCNN, EEGNet
    from pattern_recognition.data.time_shift import build_timeshifted_dataset
    from pattern_recognition.losses import GraphLoss
    assert compute_itr(0.5, 2) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/leonid/Projects/pattern_recognition && python -m pytest tests/test_imports.py -v`  
Expected: FAIL (package missing or import error)

- [ ] **Step 3: Create package layout and move code**

1. Create directories under `pattern_recognition/`.
2. Copy/move file contents:
   - `src/models_cnn.py` → `pattern_recognition/models/cnn.py` (preserve duplicate `EEGNet` definitions as-is; last definition wins — do not “fix” without documenting).
   - `src/models_gnn.py` → `pattern_recognition/models/gnn.py`
   - `src/graph.py` → `pattern_recognition/graph/electrodes.py`
   - `src/losses.py` → `pattern_recognition/losses.py`
   - `src/interpretation.py` → `pattern_recognition/interpretation.py`
   - `src/time_shift.py` → `pattern_recognition/data/time_shift.py`
   - `src/data.py` → `pattern_recognition/data/datasets.py` with imports changed to:
     ```python
     from pattern_recognition.data.p300 import P300Getter
     ```
3. Split `src/utils.py`:
   - `compute_itr` → `pattern_recognition/training/metrics.py`
   - `P300Getter`, `get_motor_subject`, `get_cursor_data`, `to_tensor`, `ddp_setup`, `count_parameters` → `pattern_recognition/data/p300.py` (or keep DDP/`count_parameters` in `training/loop.py` if cleaner — pick one place and stick to it)
   - `train_model`, `validate_model`, `infer_model`, `plot_sample`, `show_progress`, `paired_proportions_exact_test` → `pattern_recognition/training/loop.py`
4. Fix internal imports to absolute package form, e.g. in `loop.py`:
   ```python
   from pattern_recognition.losses import GraphLoss
   from pattern_recognition.training.metrics import compute_itr
   ```
5. Add empty `__init__.py` files under every package directory.
6. Update `pyproject.toml`:
   ```toml
   pydantic = "^2.0.0"
   ```
   and
   ```toml
   [tool.poetry.group.dev.dependencies]
   pytest = "^8.0.0"
   ```
7. `poetry install` (or `pip install -e ".[dev]"` if poetry groups need wiring).

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_imports.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition tests/test_imports.py pyproject.toml poetry.lock
git commit -m "feat: scaffold pattern_recognition package and move core modules"
```

---

### Task 2: Device resolution + format tests

**Files:**
- Create: `pattern_recognition/training/device.py`
- Create: `tests/test_device.py`
- Create: `tests/test_metrics_itr.py`

**Interfaces:**
- Produces:
  ```python
  def resolve_device(requested: str) -> tuple[str, "torch.device"]:
      """Return (device_resolved_str, torch.device).
      requested in {"auto","cpu","cuda","cuda:N"}.
      Explicit cuda* raises RuntimeError if unavailable.
      """
  ```

- [ ] **Step 1: Write failing tests**

```python
# tests/test_device.py
import pytest
import torch
from pattern_recognition.training.device import resolve_device

def test_resolve_cpu():
    resolved, dev = resolve_device("cpu")
    assert resolved == "cpu"
    assert dev.type == "cpu"

def test_resolve_auto_returns_string_and_device():
    resolved, dev = resolve_device("auto")
    assert resolved in {"cpu", "cuda", "cuda:0"} or resolved.startswith("cuda")
    assert isinstance(dev, torch.device)

def test_explicit_cuda_fails_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_device("cuda")
```

```python
# tests/test_metrics_itr.py
from pattern_recognition.training.metrics import compute_itr

def test_itr_perfect_binary():
    assert compute_itr(1.0, 2) > 0

def test_itr_chance_near_zero():
    assert abs(compute_itr(0.5, 2)) < 1e-6
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `poetry run pytest tests/test_device.py tests/test_metrics_itr.py -v`  
Expected: FAIL on missing `resolve_device` (ITR may already pass)

- [ ] **Step 3: Implement `resolve_device`**

```python
# pattern_recognition/training/device.py
from __future__ import annotations
import torch

def resolve_device(requested: str) -> tuple[str, torch.device]:
    requested = requested.strip().lower()
    if requested == "cpu":
        return "cpu", torch.device("cpu")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0", torch.device("cuda:0")
        return "cpu", torch.device("cpu")
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{requested}' but CUDA is unavailable. "
                "Use 'cpu' or 'auto'."
            )
        if requested == "cuda":
            return "cuda:0", torch.device("cuda:0")
        idx = int(requested.split(":", 1)[1])
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {requested} but only {torch.cuda.device_count()} CUDA device(s)."
            )
        return f"cuda:{idx}", torch.device(f"cuda:{idx}")
    raise ValueError(
        f"Unknown device '{requested}'. Expected auto|cpu|cuda|cuda:N."
    )
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `poetry run pytest tests/test_device.py tests/test_metrics_itr.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/training/device.py tests/test_device.py tests/test_metrics_itr.py
git commit -m "feat: add device resolution with explicit cuda hard-fail"
```

---

### Task 3: Pipeline + model registries

**Files:**
- Create: `pattern_recognition/data/pipelines/base.py`
- Create: `pattern_recognition/data/pipelines/registry.py`
- Create: `pattern_recognition/data/pipelines/__init__.py` (import pipeline modules for side-effect registration)
- Create: `pattern_recognition/data/pipelines/synthetic.py` (register `SyntheticBinary` for tests)
- Create: `pattern_recognition/models/registry.py`
- Create: `pattern_recognition/models/__init__.py` (register CNN models)
- Create: `tests/test_registries.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class DatasetBundle:
      train: Any  # torch Dataset or (X,y)
      val: Any
      test: Any | None
      metadata: dict

  class DataPipeline(Protocol):
      def build(self) -> DatasetBundle: ...

  def register_pipeline(name: str): ...
  def get_pipeline(name: str) -> type: ...
  def list_pipelines() -> list[str]: ...

  def register_model(name: str): ...
  def get_model(name: str) -> type: ...
  def list_models() -> list[str]: ...
  ```
- Model factory convention: `get_model(name)(**params)` constructs `nn.Module`. For `EEGNet`, accept config alias `n_channels` → map to `in_channels` in a thin wrapper or in `get_model` kwargs normalization.

- [ ] **Step 1: Write failing registry tests**

```python
# tests/test_registries.py
import pytest
from pattern_recognition.data.pipelines.registry import get_pipeline, list_pipelines
from pattern_recognition.models.registry import get_model, list_models

def test_synthetic_pipeline_registered():
    assert "SyntheticBinary" in list_pipelines()
    pipe = get_pipeline("SyntheticBinary")(n_train=32, n_val=16, n_channels=1, n_times=64)
    bundle = pipe.build()
    assert bundle.train is not None and bundle.val is not None

def test_unknown_pipeline_lists_options():
    with pytest.raises(KeyError, match="SyntheticBinary"):
        get_pipeline("DoesNotExist")

def test_eegnet_and_basecnn_registered():
    assert "EEGNet" in list_models()
    assert "BaseCNN" in list_models()
    m = get_model("EEGNet")(input_feat_dim=64, n_channels=1)
    assert m is not None

def test_unknown_model_lists_options():
    with pytest.raises(KeyError, match="EEGNet"):
        get_model("Nope")
```

- [ ] **Step 2: Run — expect FAIL**

Run: `poetry run pytest tests/test_registries.py -v`

- [ ] **Step 3: Implement registries + SyntheticBinary + register EEGNet/BaseCNN**

`SyntheticBinary.build()` returns random `torch.utils.data.TensorDataset` (or `CNNMatrixDataset`) with one-hot or float labels compatible with existing `train_model` CNN path.

Register at import time:

```python
@register_model("EEGNet")
def build_eegnet(**params):
    if "n_channels" in params and "in_channels" not in params:
        params = {**params, "in_channels": params.pop("n_channels")}
    else:
        params = {k: v for k, v in params.items() if k != "n_channels"}
    from pattern_recognition.models.cnn import EEGNet
    return EEGNet(**params)

@register_model("BaseCNN")
def build_basecnn(**params):
    # map n_channels if present
    ...
    from pattern_recognition.models.cnn import BaseCNN
    return BaseCNN(**params)
```

Ensure `pattern_recognition.data.pipelines` and `pattern_recognition.models` `__init__.py` import registration modules so `list_*` is populated on package import.

- [ ] **Step 4: Run — expect PASS**

Run: `poetry run pytest tests/test_registries.py -v`

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/data/pipelines pattern_recognition/models/registry.py pattern_recognition/models/__init__.py tests/test_registries.py
git commit -m "feat: add pipeline and model registries with synthetic pipeline"
```

---

### Task 4: Experiment schema + runner + artifact contract

**Files:**
- Create: `pattern_recognition/experiment/schema.py`
- Create: `pattern_recognition/experiment/runner.py`
- Create: `pattern_recognition/experiment/__init__.py` (export `run_experiment`)
- Create: `pattern_recognition/experiment/__main__.py`
- Create: `tests/test_schema.py`
- Create: `tests/test_run_artifacts.py`
- Create: `configs/synthetic_smoke.json`
- Modify: `.gitignore` (create if missing) to ignore `results/`

**Interfaces:**
- Produces:
  ```python
  class ExperimentConfig(BaseModel):  # pydantic v2
      name: str
      seed: int = 42
      device: str = "auto"
      data: DataConfig  # pipeline: str, params: dict
      model: ModelConfig  # name: str, params: dict
      train: TrainConfig  # lr, weight_decay, batch_size, num_epochs, step_size, gamma, save_model
      output_dir: str = "results/"

  def run_experiment(config: ExperimentConfig | dict | str | Path) -> Path:
      """Validate, run, write artifacts, return run directory Path."""
  ```
- Artifact contract keys for `metrics.json` (required):  
  `accuracy`, `balanced_accuracy`, `f1`, `itr`, `train_time_sec`, `device_requested`, `device_resolved`  
- `run_meta.json` required: `seed`, `started_at`, `finished_at`, `device_requested`, `device_resolved`, `name`

- [ ] **Step 1: Write failing schema + artifact tests**

```python
# tests/test_schema.py
import pytest
from pattern_recognition.experiment.schema import ExperimentConfig

VALID = {
    "name": "smoke",
    "seed": 0,
    "device": "cpu",
    "data": {"pipeline": "SyntheticBinary", "params": {"n_train": 32, "n_val": 16, "n_channels": 1, "n_times": 64}},
    "model": {"name": "BaseCNN", "params": {"input_feat_dim": 64, "n_channels": 1}},
    "train": {"lr": 1e-3, "weight_decay": 0.0, "batch_size": 8, "num_epochs": 1, "step_size": 1, "gamma": 1.0, "save_model": False},
    "output_dir": "results/",
}

def test_valid_config_parses():
    cfg = ExperimentConfig.model_validate(VALID)
    assert cfg.name == "smoke"

def test_invalid_device_rejected():
    bad = {**VALID, "device": "tpu"}
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(bad)
```

```python
# tests/test_run_artifacts.py
from pathlib import Path
import json
from pattern_recognition.experiment import run_experiment

REQUIRED_METRICS = {"accuracy", "balanced_accuracy", "f1", "itr", "train_time_sec", "device_requested", "device_resolved"}
REQUIRED_META = {"seed", "started_at", "finished_at", "device_requested", "device_resolved", "name"}

def test_run_experiment_writes_contract(tmp_path):
    cfg = {
        "name": "smoke",
        "seed": 0,
        "device": "cpu",
        "data": {"pipeline": "SyntheticBinary", "params": {"n_train": 32, "n_val": 16, "n_channels": 1, "n_times": 64}},
        "model": {"name": "BaseCNN", "params": {"input_feat_dim": 64, "n_channels": 1}},
        "train": {"lr": 1e-3, "weight_decay": 0.0, "batch_size": 8, "num_epochs": 1, "step_size": 1, "gamma": 1.0, "save_model": True},
        "output_dir": str(tmp_path),
    }
    run_dir = run_experiment(cfg)
    assert (run_dir / "config.json").exists()
    assert (run_dir / "run_meta.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "history.npz").exists()
    assert (run_dir / "model.pt").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert REQUIRED_METRICS <= set(metrics)
    assert REQUIRED_META <= set(meta)
    assert metrics["device_resolved"] == "cpu"
```

- [ ] **Step 2: Run — expect FAIL**

Run: `poetry run pytest tests/test_schema.py tests/test_run_artifacts.py -v`

- [ ] **Step 3: Implement schema + runner**

Runner algorithm:
1. Parse `ExperimentConfig` from path/dict/model.
2. `device_requested = cfg.device`; `device_resolved, device = resolve_device(...)`.
3. Seed numpy/torch/random.
4. `bundle = get_pipeline(cfg.data.pipeline)(**cfg.data.params).build()`.
5. Build DataLoaders from bundle (batch_size from train).
6. `model = get_model(cfg.model.name)(**cfg.model.params)`.
7. Train with existing `train_model` (adapt learning_params dict: include `model_type="CNN"`, lr, weight_decay, step_size, gamma, num_epochs). Use MSE or whatever the loop already expects for CNN binary one-hot labels — match `CNNMatrixDataset` label format.
8. Write `results/<name>_<YYYYMMDD_HHMMSS>/` with all artifacts.
9. Return `Path`.

CLI `__main__.py`:
```python
import argparse, sys
from pattern_recognition.experiment.runner import run_experiment

def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("config")
    args = p.parse_args(argv)
    if args.cmd == "run":
        path = run_experiment(args.config)
        print(path)

if __name__ == "__main__":
    main()
```

Add `configs/synthetic_smoke.json` matching the test config (paths relative).

- [ ] **Step 4: Run — expect PASS**

Run: `poetry run pytest tests/test_schema.py tests/test_run_artifacts.py -v`  
Also: `poetry run python -m pattern_recognition.experiment run configs/synthetic_smoke.json`

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/experiment configs/synthetic_smoke.json tests/test_schema.py tests/test_run_artifacts.py .gitignore
git commit -m "feat: add experiment schema, runner, and artifact contract"
```

---

### Task 5: Real data pipelines (Samara average, time-shift, BCI3 Pz)

**Files:**
- Create: `pattern_recognition/data/pipelines/samara_average.py`
- Create: `pattern_recognition/data/pipelines/samara_time_shift.py`
- Create: `pattern_recognition/data/pipelines/bci3_pz.py`
- Create: `pattern_recognition/data/averaging.py` (pure helpers extracted from `epoch_averaging.ipynb`: `build_multichannel_subject_dataset_unique`, `multichannel_to_single_channel`, per-sample standardize)
- Create: `tests/test_averaging_shapes.py`
- Create: `tests/test_time_shift_math.py`
- Create: `configs/samara_pz_eegnet_sc_n10.json`
- Create: `configs/samara_pz_basecnn_sc_n10.json`
- Modify: `pattern_recognition/data/pipelines/__init__.py` to import new pipelines

**Interfaces:**
- Produces registered pipelines:
  - `SamaraWithinSubjectAverage(path, channel_idx=1, n_average=10, mode="SC"|"MC", file_pattern=..., val_fraction=0.2, seed=...)`
  - `SamaraTimeShift(path, channel_idx=1, shift_ms=100, n_channels=3, epoch_len=250, fs=250, val_fraction=0.2, seed=...)`
  - `BCI3PzEpochAverage(train_mat, test_mat=None, channel_name="Pz", n_average=..., mode=..., eloc montage via mne, ...)` — wrap `P300Getter` + averaging helpers; params must match what matrix notebooks need at minimum for a runnable config.
- Averaging helpers operate on numpy/torch and are unit-tested without `.mat` files.

- [ ] **Step 1: Write failing unit tests (no real data required)**

```python
# tests/test_time_shift_math.py
import numpy as np
from pattern_recognition.data.time_shift import build_timeshifted_dataset, time_shift_info

def test_timeshift_shape_and_bounds():
    raw = np.random.randn(20, 500).astype(np.float32)
    y = np.zeros(20, dtype=np.int64)
    X, yy = build_timeshifted_dataset(raw, y, shift_ms=100, n_channels=3, epoch_len=250, fs=250)
    assert X.shape == (20, 3, 250)
    info = time_shift_info(500, 250, 250, 100, 3)
    assert info["fits_in_raw"] is True

def test_timeshift_raises_if_too_long():
    raw = np.random.randn(5, 500).astype(np.float32)
    y = np.zeros(5, dtype=np.int64)
    import pytest
    with pytest.raises(ValueError):
        build_timeshifted_dataset(raw, y, shift_ms=100, n_channels=20, epoch_len=250, fs=250)
```

```python
# tests/test_averaging_shapes.py
import torch
from pattern_recognition.data.averaging import (
    build_multichannel_subject_dataset_unique,
    multichannel_to_single_channel,
)

def test_mc_and_sc_shapes():
    data = torch.randn(100, 250)
    labels = torch.tensor([1] * 50 + [0] * 50)
    X, y = build_multichannel_subject_dataset_unique(data, labels, n_channels=5, seed=0)
    assert X.ndim == 3 and X.shape[1] == 5 and X.shape[2] == 250
    X_sc = multichannel_to_single_channel(X)
    assert X_sc.shape[1] == 1
```

- [ ] **Step 2: Run — expect FAIL**

Run: `poetry run pytest tests/test_time_shift_math.py tests/test_averaging_shapes.py -v`

- [ ] **Step 3: Implement helpers + three pipelines**

Port averaging logic from `multi_eeg_notebooks/epoch_averaging.ipynb` (functions already identified).  
`SamaraWithinSubjectAverage`: use `load_p300_subjects` from `time_shift` (or shared loader), build MC then optionally SC, split train/val. Raise `FileNotFoundError` with resolved path if `path` missing.  
`SamaraTimeShift`: `load_p300_subjects` + `build_timeshifted_dataset` (+ SC mean option via param `mode` if useful).  
`BCI3PzEpochAverage`: load BCI III `.mat` via existing `P300Getter` flow; extract single channel Pz; apply same averaging helpers; document required params in docstring.

Register all three names exactly as in the spec.

Example configs point at `Samara_data/` and use EEGNet/BaseCNN SC n=10 (for users who have data).

- [ ] **Step 4: Run unit tests — expect PASS**

Run: `poetry run pytest tests/test_time_shift_math.py tests/test_averaging_shapes.py tests/test_registries.py -v`  
Expected: PASS; real Samara/BCI runs are manual if data present:

```bash
# only if Samara_data/ exists
poetry run python -m pattern_recognition.experiment run configs/samara_pz_eegnet_sc_n10.json
```

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/data/averaging.py pattern_recognition/data/pipelines configs/samara_*.json tests/test_averaging_shapes.py tests/test_time_shift_math.py
git commit -m "feat: add Samara/BCI3 data pipelines and averaging helpers"
```

---

### Task 6: Reporting module

**Files:**
- Create: `pattern_recognition/reporting/load.py`
- Create: `pattern_recognition/reporting/tables.py`
- Create: `pattern_recognition/reporting/plots.py`
- Create: `pattern_recognition/reporting/__init__.py`
- Create: `tests/test_reporting.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class RunArtifacts:
      path: Path
      config: dict
      meta: dict
      metrics: dict
      history: dict  # arrays from npz

  def load_run(path: str | Path) -> RunArtifacts: ...
  def metrics_table(run_dirs: list[str | Path]) -> "pd.DataFrame": ...
  def plot_training_curves(run: RunArtifacts, ax=None): ...
  def compare_runs(run_dirs: list[str | Path]): ...  # curves +/or metric bars; show or return figs
  ```
- `metrics_table` columns at least: `name`, `device_resolved`, `accuracy`, `balanced_accuracy`, `f1`, `itr`, `train_time_sec`, `model.name`, `data.pipeline`, and `mode` / `n_average` when present in config.

- [ ] **Step 1: Write failing reporting test using synthetic runs**

```python
# tests/test_reporting.py
from pattern_recognition.experiment import run_experiment
from pattern_recognition.reporting import load_run, metrics_table

def _cfg(name, tmp_path):
    return {
        "name": name,
        "seed": 0,
        "device": "cpu",
        "data": {"pipeline": "SyntheticBinary", "params": {"n_train": 32, "n_val": 16, "n_channels": 1, "n_times": 64}},
        "model": {"name": "BaseCNN", "params": {"input_feat_dim": 64, "n_channels": 1}},
        "train": {"lr": 1e-3, "weight_decay": 0.0, "batch_size": 8, "num_epochs": 1, "step_size": 1, "gamma": 1.0, "save_model": False},
        "output_dir": str(tmp_path),
    }

def test_metrics_table_two_runs(tmp_path):
    d1 = run_experiment(_cfg("run_a", tmp_path))
    d2 = run_experiment(_cfg("run_b", tmp_path))
    run = load_run(d1)
    assert "accuracy" in run.metrics
    table = metrics_table([d1, d2])
    assert len(table) == 2
    assert "name" in table.columns and "device_resolved" in table.columns
```

- [ ] **Step 2: Run — expect FAIL**

Run: `poetry run pytest tests/test_reporting.py -v`

- [ ] **Step 3: Implement load / tables / plots**

Use pandas + matplotlib. `compare_runs` should at least plot accuracy bars and overlay loss curves if `history` has loss.

- [ ] **Step 4: Run — expect PASS**

Run: `poetry run pytest tests/test_reporting.py -v`

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/reporting tests/test_reporting.py
git commit -m "feat: add reporting loaders, metrics table, and plots"
```

---

### Task 7: Example notebook + README + remove `src/`

**Files:**
- Create: `notebooks/examples/run_experiment_samara.ipynb`
- Modify: `README.md` (full rewrite per spec README content)
- Delete: `src/` (entire tree) once tests pass against the package
- Modify: any docs that only mention `src/` if needed (optional)

**Interfaces:**
- Notebook demonstrates: import package → run two configs (or SyntheticBinary twice / EEGNet+BaseCNN if data available) → `load_run` → `metrics_table` → `compare_runs` → Colab section cells from the spec.
- README sections: overview, datasets, experiment setups, models, metrics, quickstart (local+Colab), repo map.

- [ ] **Step 1: Write README** following the spec’s “README content” section (tables for datasets; link `configs/`; explain MC/SC, time-shift, BCI3; metrics list; quickstart commands).

- [ ] **Step 2: Create example notebook** with cells (markdown + code) — no need to execute heavy training in CI. Include Colab install/mount/run/compare cells from the design spec.

- [ ] **Step 3: Delete `src/` and run full test suite**

```bash
rm -rf src
poetry run pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md notebooks/examples/run_experiment_samara.ipynb
git add -u src
git commit -m "docs: research README, example notebook; remove legacy src/"
```

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|---|---|
| Installable `pattern_recognition` package + absolute imports | 1 |
| Split utils; preserve model math | 1 |
| Device auto/cpu/cuda + hard-fail + record both | 2, 4 |
| Pipeline registry + 3 real pipelines + synthetic for tests | 3, 5 |
| Model registry EEGNet/BaseCNN | 3 |
| Pydantic config + run artifacts + CLI | 4 |
| Reporting load/table/plots + comparison | 6 |
| Format/schema/registry/unit/device/smoke tests | 2–6 |
| Example notebook + Colab + metrics comparison | 7 |
| README research entry point | 7 |
| Delete `src/` | 7 |
| pydantic + pytest deps | 1 |

No reporting CLI (explicitly out of scope). GNN runner deferred (models still importable).

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-08-08-package-and-experiment-runner.md`.
