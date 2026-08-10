# pattern_recognition

Research code for P300 / EEG pattern recognition with two parallel tracks:

- **GNN track** — multi-channel EEG as a graph (electrode topology + geometric DL); BCI Competition III P300 (64 ch) and SEED emotion recognition.
- **Epoch-averaging track** (v1 experiment runner) — raise SNR by averaging trials; Pz / MC–SC protocols; CNN (+ SVM baselines in historical notebooks) on Samara and BCI III.

The installable package is `pattern_recognition`. Experiments for the epoch-averaging / CNN path are driven by JSON configs via `run_experiment`. Detailed design: [`docs/superpowers/specs/2026-08-08-package-and-experiment-runner-design.md`](docs/superpowers/specs/2026-08-08-package-and-experiment-runner-design.md).

## Datasets

| Dataset | Task | Typical use in this repo | Location / source |
|---|---|---|---|
| BCI Competition III, dataset II | P300 speller | GNN 64-ch; epoch-average Pz (Subjects A/B) | [bbci.de](https://www.bbci.de/competition/iii/#top); `raw_data/` |
| Samara multi-EEG (S0201…S2001) | P300 classic / Aperture B | Epoch averaging, time-shift | `Samara_data/` (and `processed_data/` CSVs) |
| SEED | 3-class emotion | GNN notebooks | documented path in SEED notebooks |

**Channel / time conventions (typical after pipeline):**

- Samara Pz: 250 samples @ 250 Hz (single-channel or MC stacks).
- BCI III: 64 channels, 72 samples @ 120 Hz after the GNN / matrix pipelines; Pz epoch-average path uses the Pz channel only.

## Experiment setups

| Setup | Modes / notes | Config / entry |
|---|---|---|
| Within-subject epoch averaging | **MC** (N epochs → N channels) vs **SC** (average → 1 channel); typical N=5/10 | [`configs/samara_pz_eegnet_sc_n10.json`](configs/samara_pz_eegnet_sc_n10.json), [`configs/samara_pz_basecnn_sc_n10.json`](configs/samara_pz_basecnn_sc_n10.json); pipeline `SamaraWithinSubjectAverage` |
| Cross-subject / mixed / K-trials | Historical protocols in notebooks | `multi_eeg_notebooks/epoch_averaging.ipynb` |
| Time-shifted multi-channel windows | Long epochs → shifted windows | pipeline `SamaraTimeShift` |
| BCI III Pz epoch-average | Subjects A/B validation | pipeline `BCI3PzEpochAverage` |
| Synthetic smoke / CI | Tiny binary tensors | [`configs/synthetic_smoke.json`](configs/synthetic_smoke.json); pipeline `SyntheticBinary` |
| GNN static / edge-learning graphs | Delaunay / k-NN / prior; edge-learn models | `notebooks/` (runner support not in v1) |

Example notebook (run → load → compare): [`notebooks/examples/run_experiment_samara.ipynb`](notebooks/examples/run_experiment_samara.ipynb).

## Speller benchmark

Character-level accuracy and ITR (vs flash repetition count) from a trained binary P300 model. Design: [`docs/superpowers/specs/2026-08-09-bci-speller-benchmark-design.md`](docs/superpowers/specs/2026-08-09-bci-speller-benchmark-design.md).

By default (`use_synthetic: false`) selections come from **real EEG** via binary `data.params` and/or speller `protocol_params`:
- BCI III — `test_mat` + `eloc_path` + `StimulusCode` (ground truth)
- Samara — `path` to `.mat` dir + holdout + 4×4 simulation (`label_source: simulated`)

Set `use_synthetic: true` only for CI smoke (random flashes).

```bash
# after a binary run exists under results/<name>_<timestamp>/
python -m pattern_recognition.speller run \
  --config configs/speller_bci3_within.json \
  --run-dir results/<binary_run>/
```

Example configs: [`configs/speller_bci3_within.json`](configs/speller_bci3_within.json) (BCI III row×column), [`configs/speller_samara_sim_within.json`](configs/speller_samara_sim_within.json) (Samara simulated protocol). Artifacts land in `results/<binary_run>/speller/<tag>/`.

## Models

| Model | Role |
|---|---|
| `EEGNet` | Compact CNN baseline for P300 / epoch-averaged windows (runner registry) |
| `BaseCNN` | Simple 1D-CNN baseline (runner registry) |
| `DeepConvNet`, `CecottiCNN`, `BaseCNNAttn`, `FlexCNN` | Additional CNN variants (importable; not all registered for the runner) |
| SVM (RBF) | Classical baseline in historical epoch-averaging notebooks |
| `BaseGNN`, `GIN`, `STGCN`, … | Static-graph GNN family for 64-ch / SEED notebooks |
| `EdgeLearnGNN`, `PriorEdgeLearnGNN`, `PairEdgeLearnGNN`, … | Adaptive / learned-edge GNNs |

Runner v1 registers **`EEGNet`** and **`BaseCNN`**. GNN models remain importable for notebooks.

## Metrics

Reported in run `metrics.json` (and comparison tables):

- **Accuracy** (and CI where used in notebooks)
- **Balanced accuracy** — important under P300 class imbalance
- **F1**
- **ITR** (Wolpaw, bits/trial) via `pattern_recognition.training.metrics.compute_itr`
- **Device fields** — `device_requested` and `device_resolved` in run artifacts

Compare saved runs:

```python
from pattern_recognition.reporting import metrics_table, compare_runs

table = metrics_table(run_dirs)
compare_runs(run_dirs)
```

## Quickstart

### Local

```bash
poetry install
# or: pip install -e .

# smoke (synthetic data, no Samara/BCI files required)
python -m pattern_recognition.experiment run configs/synthetic_smoke.json

# Samara Pz SC N=10 (requires Samara_data/)
python -m pattern_recognition.experiment run configs/samara_pz_eegnet_sc_n10.json
python -m pattern_recognition.experiment run configs/samara_pz_basecnn_sc_n10.json
```

Artifacts land under `results/<name>_<timestamp>/` (`config.json`, `run_meta.json`, `metrics.json`, `history.npz`, optional `model.pt`).

Programmatic:

```python
from pattern_recognition.experiment import run_experiment
from pattern_recognition.reporting import load_run, metrics_table, compare_runs

run_dir = run_experiment("configs/synthetic_smoke.json")
run = load_run(run_dir)
print(run.metrics)
```

### Colab

```python
!pip install -q "git+https://github.com/Sidl419/pattern_recognition.git"
# or: %cd /content/drive/MyDrive/pattern_recognition && !pip install -q -e .

from google.colab import drive
drive.mount("/content/drive")

from pattern_recognition.experiment import run_experiment

config = {
    "name": "colab_samara_eegnet_sc",
    "seed": 42,
    "device": "cuda",  # or "auto" / "cpu"
    "data": {
        "pipeline": "SamaraWithinSubjectAverage",
        "params": {
            "path": "/content/drive/MyDrive/Samara_data/",
            "channel_idx": 1,
            "n_average": 10,
            "mode": "SC",
        },
    },
    "model": {
        "name": "EEGNet",
        "params": {"n_channels": 1, "input_feat_dim": 250},
    },
    "train": {
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "batch_size": 64,
        "num_epochs": 50,
        "step_size": 20,
        "gamma": 0.5,
        "save_model": True,
    },
    "output_dir": "/content/drive/MyDrive/pattern_recognition_results/",
}

run_dir = run_experiment(config)
```

Point `data.params.path` / `output_dir` at Drive mounts. Use `"device": "cuda"` on GPU runtimes (hard-fails if no GPU); `"auto"` is fine too. See the Colab section in the example notebook for mount / compare cells.

## Repo map

```text
pattern_recognition/   # installable package (data, models, training, experiment, reporting)
configs/               # example JSON experiment configs
notebooks/examples/    # thin runner + reporting demo
notebooks/             # historical GNN / SEED research notebooks
multi_eeg_notebooks/   # historical epoch-averaging notebooks
results/               # run outputs (gitignored)
docs/                  # design specs and plans
tests/                 # pytest (schema, registries, unit, smoke)
```

## Tests

```bash
poetry run pytest tests/ -v
```

GitHub Actions runs the same suite on push/PR to `main` (CPU Torch; `torch-geometric` is best-effort so CNN/unit tests still run if PyG wheels fail).
