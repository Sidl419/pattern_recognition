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
- **Experiments:** config → `run_experiment` → `results/<name>_<timestamp>/`. Reporting reads run dirs only (`load_run`, `metrics_table`, `compare_runs`). Run dirs are never reused; config/meta are written before training so failures leave a record.
- **Configs are strict:** all schema models use `extra="forbid"`. Do not loosen this — a silently ignored key is a silently different experiment. Set a split seed in one place only (`split.seed` *or* `data.params.seed`).
- **Speller holdout:** Samara character evaluation reads `split_indices.json` from the binary run. Never recompute the epoch split on the speller side.
- **Per-subject seeds:** always `data.splits.subject_seed(base_seed, subject_id)`. Never `base_seed + i` over an enumerated subject list — the split then depends on cohort membership and ordering.
- **Samara loading/splitting:** go through `data.samara` (`load_samara_subjects`, `subject_epoch_splits`, `as_split_indices`). Do not re-inline the load/filter/split block in a new pipeline.
- **Layering:** `selections/` (grids, protocols, packing, decode) is the shared leaf; `data/` may import it but must never import `speller/`. `tests/test_layering.py` enforces this — fix the design, not the test.
- **Reported metrics** come from `best_epoch` and the saved checkpoint must match them. Keep `metrics.json` and `model.pt` in agreement. Default `train.checkpoint_metric` is `brier` (lower is better).
- **CNN models emit logits.** `BrierLoss` applies the softmax. Never add a `Sigmoid`/`Softmax` to a CNN `forward` in `models/cnn.py` — that double-applies the link and silently changes the objective. The Brier score (not cross-entropy) is the deliberate choice for this data's imbalance. GNN notebooks are a separate track and still apply their own output activations.
- **Device:** `auto` | `cpu` | `cuda` | `cuda:N`. Explicit CUDA hard-fails if unavailable. Record `device_requested` and `device_resolved`.
- **Model math:** do not change architectures or default hyperparameters unless fixing a clear bug (document it).
- **Notebooks:** prefer `notebooks/examples/` for the new workflow. Do not clear historical notebook outputs unless the user asks.
- **Data:** do not commit `Samara_data/`, `raw_data/`, `processed_data/`, or large `.mat` / CSV dumps. `results/` is gitignored.

## Where things live

| Area | Path |
|---|---|
| Package | `pattern_recognition/` |
| Shared paradigm layer | `pattern_recognition/selections/` |
| Samara load/split helpers | `pattern_recognition/data/samara.py` |
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
