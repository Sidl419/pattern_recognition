# Package restructure + experiment runner design

**Date:** 2026-08-08  
**Status:** Draft for user review  
**Scope:** Installable `pattern_recognition` package, code-style cleanup, config-driven experiment runner (epoch-averaging / CNN track), reporting helpers, format tests. Notebook import cells updated without clearing outputs.

## Goals

1. Replace flat `src/*.py` imports with an installable package matching Poetry (`pattern_recognition`).
2. Improve structure and code style without changing scientific behavior of existing models/training.
3. Run experiments from JSON configs with minimal duplication (pipeline + model registries).
4. Save reproducible run artifacts; support plots/tables via a reporting module that reads those artifacts.
5. Keep Google Colab usable (`pip install -e .` / git install + path config).
6. Preserve notebook cell outputs when updating imports.

## Non-goals (v1)

- Migrating GNN / SEED training into the experiment runner (package must still expose existing GNN models for notebooks).
- Re-running notebooks or regenerating their outputs.
- Full hyperparameter search / multi-GPU DDP rewrite (keep existing DDP helpers if present; not required for v1 runner).
- Pulling all duplicated notebook logic into the package in one pass (only what the epoch-averaging runner needs).

## Package layout

```text
pattern_recognition/
  __init__.py
  data/
    __init__.py
    p300.py              # P300Getter (from utils)
    datasets.py          # CNNMatrixDataset, GraphMatrixDataset, EEGDataset
    pipelines/           # DataPipeline implementations + registry
      __init__.py
      base.py
      registry.py
      samara_average.py
      samara_time_shift.py
      bci3_pz.py         # BCI III Pz epoch-average pipeline
  models/
    __init__.py
    cnn.py
    gnn.py
    registry.py
  graph/
    __init__.py
    electrodes.py        # Delaunay / k-NN / prior graph helpers
  training/
    __init__.py
    metrics.py           # ITR, binary metric helpers
    loop.py              # train_model, validate_model, infer_model
    device.py            # resolve device: auto | cpu | cuda | cuda:N
  reporting/
    __init__.py
    load.py              # load_run, list artifacts
    tables.py            # metrics tables / compare runs
    plots.py             # training curves, comparison plots
  experiment/
    __init__.py
    schema.py            # Pydantic config models
    runner.py            # run_experiment
    __main__.py          # CLI: python -m pattern_recognition.experiment run <config.json>
  losses.py
  interpretation.py
configs/                 # checked-in example JSON configs
results/                 # run outputs (gitignore)
tests/                   # pytest: schema/format, registry, unit, optional smoke
```

`pyproject.toml` already declares `packages = [{include = "pattern_recognition"}]`. After the move, remove or replace flat `src/` (prefer delete once notebooks import the package; no long-lived shims unless a short transition is needed).

### Style / revision rules

- Relative imports inside the package; no `from utils import ...`.
- Type hints on public APIs; docstrings on pipelines and runner entrypoints.
- Split kitchen-sink `utils.py` into `data/`, `training/`, `metrics`.
- Deduplicate near-identical `CNNMatrixDataset` / `EEGDataset` only if behavior stays identical.
- Do not change model math or default hyperparameters unless fixing a clear bug (document any intentional fix).

### Notebooks

- Update import cells only: `from pattern_recognition...`.
- Do not clear or rewrite outputs.
- Prefer EditNotebook / surgical JSON edits that leave `outputs` arrays untouched.

## Experiment config

### Top-level shape

```json
{
  "name": "samara_pz_eegnet_sc_n10",
  "seed": 42,
  "device": "auto",
  "data": {
    "pipeline": "SamaraWithinSubjectAverage",
    "params": {
      "path": "Samara_data/",
      "channel_idx": 1,
      "n_average": 10,
      "mode": "SC"
    }
  },
  "model": {
    "name": "EEGNet",
    "params": {
      "n_channels": 1,
      "input_feat_dim": 250
    }
  },
  "train": {
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "batch_size": 64,
    "num_epochs": 50,
    "step_size": 20,
    "gamma": 0.5,
    "save_model": true
  },
  "output_dir": "results/"
}
```

### Device policy

| Config value | Behavior |
|---|---|
| `"auto"` | Use CUDA if available, else CPU |
| `"cpu"` | Force CPU |
| `"cuda"` / `"cuda:N"` | Require that device; if unavailable, **fail with a clear error** (no silent CPU fallback) |

Every run records both:

- `device_requested` — string from config  
- `device_resolved` — actual `torch.device` string used (e.g. `cuda:0`, `cpu`)

written into run artifacts (see below).

### Data pipelines (registry)

Shared interface:

```python
class DataPipeline(Protocol):
    def build(self) -> DatasetBundle:
        """Return train/val/(optional test) datasets or loaders + metadata."""
        ...
```

Config names a registered class:

```json
"data": {
  "pipeline": "SamaraWithinSubjectAverage",
  "params": { "...": "pipeline kwargs only" }
}
```

Runner: `get_pipeline(name)(**params).build()`.

**v1 pipelines (epoch-averaging / CNN track):**

| Registry name | Priority | Responsibility |
|---|---|---|
| `SamaraWithinSubjectAverage` | Required | Samara `.mat` → within-subject averaging MC/SC |
| `SamaraTimeShift` | Required | Long epochs → time-shifted multi-channel windows (`time_shift`) |
| `BCI3PzEpochAverage` | Required | BCI Competition III Pz path used by matrix / Subject A–B notebooks |

Adding a new technique = new pipeline class + `@register_pipeline(...)`. No change to runner core.

**Dependency:** add `pydantic` (and `pytest` as a dev dependency) to `pyproject.toml`.

### Model registry

Same pattern: `"model": {"name": "EEGNet", "params": {...}}` → `get_model(name)(**params)`.

v1 models: CNN track used by epoch averaging (`EEGNet`, `BaseCNN`, and others already in `models_cnn.py` as needed). GNN models remain importable for notebooks but are not required in the runner registry for v1.

## Run flow and artifacts

```text
load + validate config (Pydantic)
  → resolve device (record requested + resolved)
  → set seed
  → pipeline.build()
  → build model
  → train / validate (training.loop)
  → write results/<name>_<timestamp>/
```

**Run directory contents:**

| File | Purpose |
|---|---|
| `config.json` | Exact config used |
| `run_meta.json` | seed, timestamps, `device_requested`, `device_resolved`, package/git version if available |
| `metrics.json` | Final metrics (accuracy, balanced accuracy, F1, ITR, timing, etc.) + device fields |
| `history.npz` | Loss / metric curves |
| `model.pt` | Optional weights if `train.save_model` |

CLI:

```bash
python -m pattern_recognition.experiment run configs/foo.json
```

Programmatic (notebooks / Colab):

```python
from pattern_recognition.experiment import run_experiment
run_experiment(config_dict_or_path)
```

## Reporting

`pattern_recognition.reporting` reads **saved run directories** only:

- `load_run(path)` → config, meta, metrics, history  
- `metrics_table(run_dirs)` → comparison table (DataFrame / CSV / markdown)  
- `plot_training_curves(run)` / `compare_runs(run_dirs)` → figures  

v1 ships the Python API (`load_run`, `metrics_table`, plot helpers). A reporting CLI (`summarize`) is out of scope for v1.

Training never owns presentation; reporting never builds datasets/models.

## Colab

- Install package from repo (`pip install -e .` or `pip install git+...`).
- Point config `data.params.path` / `output_dir` at Drive mounts or uploaded paths.
- Use `"device": "cuda"` on GPU runtimes or `"auto"`.
- Same JSON + `run_experiment` API as local.

## Errors

- Unknown pipeline/model → error listing registered names.
- Invalid params → Pydantic / pipeline validation before data load.
- Missing data path → `FileNotFoundError` with resolved path.
- Explicit CUDA when unavailable → hard fail with message to switch to `cpu` or `auto`.

## Testing

| Kind | What |
|---|---|
| Format / schema | Valid sample configs parse; invalid configs rejected; `metrics.json` / run-dir contract keys present |
| Registry | Known names resolve; unknown names fail |
| Unit | Time-shift index math; ITR edge cases; averaging output shapes on synthetic arrays |
| Device | `auto` / `cpu` / invalid `cuda` resolution behavior (mock or skip if no GPU) |
| Smoke (optional, marked) | Tiny synthetic `run_experiment` end-to-end |

No full notebook execution in CI.

## Migration plan (high level)

1. Create `pattern_recognition/` package; move and split modules; fix internal imports.
2. Add device helper, registries, Pydantic schema, runner, artifact writer.
3. Implement v1 pipelines + wire CNN models into model registry.
4. Add reporting module (load + table + basic plots).
5. Add pytest format/unit tests + example configs.
6. Update notebook import cells only (preserve outputs).
7. Update `pyproject.toml` / `.gitignore` (`results/`); remove obsolete `src/` when safe.
8. Document Colab + local usage in README briefly.

## Success criteria

- `poetry install` / editable install exposes `pattern_recognition`.
- At least one real-shaped example config runs the epoch-averaging CNN path and writes artifacts including `device_resolved`.
- Format tests pass for config + metrics contracts.
- Existing notebooks import the package without losing stored outputs.
- New data technique can be added by a new pipeline class + registry entry only.
