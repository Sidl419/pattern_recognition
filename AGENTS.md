# Agent guide — pattern_recognition

Cross-tool instructions for coding agents. Prefer this file over inventing repo conventions.

## Project in one minute

Two research tracks:

1. **GNN** — multi-channel EEG as graphs (`notebooks/`, models under `pattern_recognition.models.gnn`).
2. **Epoch averaging / CNN** — SNR via trial averaging; JSON configs + `run_experiment` (v1 runner focus).
3. **Speller benchmark** — character-level evaluation from binary P300 models; code under `pattern_recognition/speller`, artifacts under `run_dir/speller/<tag>/`.

Installable package: `pattern_recognition` (Poetry). Design: `docs/superpowers/specs/2026-08-08-package-and-experiment-runner-design.md`.

## Hard rules

- **Imports:** absolute package imports only — `from pattern_recognition...`. Never flat `from utils import ...` / relative-dot imports inside the package.
- **New data technique:** add a pipeline class + `@register_pipeline("Name")`. Do not grow a fat config schema for every protocol.
- **Experiments:** config → `run_experiment` → `results/<name>_<timestamp>/`. Reporting reads run dirs only (`load_run`, `metrics_table`, `compare_runs`).
- **Device:** `auto` | `cpu` | `cuda` | `cuda:N`. Explicit CUDA hard-fails if unavailable. Record `device_requested` and `device_resolved`.
- **Model math:** do not change architectures or default hyperparameters unless fixing a clear bug (document it).
- **Notebooks:** prefer `notebooks/examples/` for the new workflow. Do not clear historical notebook outputs unless the user asks.
- **Data:** do not commit `Samara_data/`, `raw_data/`, `processed_data/`, or large `.mat` / CSV dumps. `results/` is gitignored.

## Where things live

| Area | Path |
|---|---|
| Package | `pattern_recognition/` |
| Example configs | `configs/` |
| Example notebook | `notebooks/examples/run_experiment_samara.ipynb` |
| Tests | `tests/` |
| CI | `.github/workflows/ci.yml` |

## Commands

```bash
poetry install
# optional GNN/viz: poetry install -E all
make test          # or: poetry run pytest tests/ -q
make format        # Ruff format pattern_recognition + tests
make check         # format-check + lint + test
poetry run python -m pattern_recognition.experiment run configs/synthetic_smoke.json
```

## Out of scope unless asked

- Migrating every historical notebook to the new package
- GNN experiment-runner support (models remain importable)
- Reporting CLI
