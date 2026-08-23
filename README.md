# pattern_recognition

Research code for P300 / EEG pattern recognition with two parallel tracks:

- **GNN track** — multi-channel EEG as a graph (electrode topology + geometric DL); BCI Competition III P300 (64 ch) and SEED emotion recognition.
- **Epoch-averaging track** (v1 experiment runner) — raise SNR by averaging trials; Pz / MC–SC protocols; CNN (+ SVM baselines in historical notebooks) on Samara and BCI III.

The installable package is `pattern_recognition`. Experiments for the epoch-averaging / CNN path are driven by JSON configs via `run_experiment`. Detailed design: [`docs/superpowers/specs/2026-08-08-package-and-experiment-runner-design.md`](docs/superpowers/specs/2026-08-08-package-and-experiment-runner-design.md).

## Datasets

| Dataset | Subjects | Task | Typical use in this repo | Location / source |
|---|---|---|---|---|
| BCI Competition III, dataset II | A, B | P300 row×column speller | GNN 64-ch; Pz epoch-average; speller benchmark | [bbci.de](https://www.bbci.de/competition/iii/#top); `matrix_dataset/` (also mirrored under `raw_data/`) |
| Samara multi-EEG | 10 (`S0201`…`S2001`) | P300 classic / Aperture B | Epoch averaging, time-shift, simulated speller | `Samara_data/` (`.mat`); optional `processed_data/*_PZ.csv` |
| SEED | 15 (standard release) | 3-class emotion | GNN notebooks only (not in v1 runner / speller) | path set in SEED notebooks |

### BCI Competition III — Dataset II

Classic Wadsworth P300 speller (6×6 grid, row/column intensification).

| Split | Characters | Continuous samples / char | EEG channels | Notes |
|---|---|---|---|---|
| Subject A/B **Train** | 85 each | 7794 @ native rate | 64 | Has `StimulusType` + `TargetChar` |
| Subject A/B **Test** | 100 each | 7794 | 64 | Labels withheld in the public mats; this repo uses the published target strings for eval |

After pipeline preprocessing (band-pass / resample in `P300Getter`):

- **64-ch GNN / matrix path:** epochs of **72 samples @ 120 Hz** per flash.
- **Pz epoch-average / speller path:** same flash windows, **Pz only** (plus per-sample z-score; speller also applies the shared MNE `Scaler` path).
- **Stimulus schedule:** 12 codes (6 rows + 6 columns) × **15 repeats** → **180 flashes / character** (train A: 85×180 = 15 300 flashes; test A: 100×180 = 18 000).
- Speller decode is ground-truth (`StimulusCode`); do not pool A and B in `within_subject` mode.

### Samara multi-EEG

Local collection used for within-/cross-subject P300 work and the simulated single-flash speller.

| Property | P300 classic | Aperture B |
|---|---|---|
| Subjects in this checkout | 10 (`S0201`, `S0601`, `S0701`, `S1201`, `S1401`, `S1601`, `S1701`, `S1801`, `S1901`, `S2001`) | same 10 |
| Channels in `.mat` | P4, **PZ**, P3 | same |
| Sampling rate | 250 Hz | 250 Hz |
| Raw epoch length | 500 samples (2.0 s) | 500 samples |
| Approx. epochs / subject | ~6 760 | ~6 760 |
| Class balance (P300 classic, all subjects) | ~4 215 target / ~63 402 non-target (~1:15) | similar oddball imbalance |
| Filtering recorded in mats | FIR band-pass ~1–15 Hz | same metadata fields |

**How the package uses it:**

- Default runner pipelines take **PZ** (`channel_idx=1`) and truncate to **250 samples (0–1 s @ 250 Hz)** for CNN inputs; time-shift pipelines keep the long 500-sample epochs and cut shifted 250-sample windows.
- Per-subject split seeds are derived from the **subject id** (`data.splits.subject_seed`), not the subject's position in the cohort, so a subject gets the same train/val/eval epochs whether it is trained alone or pooled with the other nine.
- Labels are binary only (`0` / `1`). There are **no stimulus IDs** in the mats, so character-level Samara evaluation is a **4×4 single-flash simulation** (`label_source: simulated`; default phrase `JUST_DO_IT`, SOA 110 ms).
- `processed_data/S*_P300_PZ.csv` / `S*_AB_PZ.csv` are optional flattened PZ dumps (~6 770×251: label + 250 samples) for notebook baselines — not required by `run_experiment`.

### SEED

SJTU Emotion EEG Dataset (standard public release): **15 subjects**, **62 channels**, **3 emotion classes** (positive / neutral / negative). Used only in historical GNN notebooks (`notebooks/`); not wired into the JSON experiment runner or speller benchmark in v1.

## Experiment setups

| Setup | Modes / notes | Config / entry |
|---|---|---|
| Within-subject epoch averaging | **MC** (N epochs → N channels) vs **SC** (average → 1 channel); typical N=5/10; fair flash-level compare uses **N=1** (`*_sc_n1`) | [`configs/samara_pz_eegnet_sc_n10.json`](configs/samara_pz_eegnet_sc_n10.json), [`configs/samara_pz_basecnn_sc_n10.json`](configs/samara_pz_basecnn_sc_n10.json), [`configs/samara_pz_eegnet_sc_n1.json`](configs/samara_pz_eegnet_sc_n1.json), [`configs/samara_pz_basecnn_sc_n1.json`](configs/samara_pz_basecnn_sc_n1.json), [`configs/samara_pz_svm_sc_n1.json`](configs/samara_pz_svm_sc_n1.json); pipeline `SamaraWithinSubjectAverage` |
| Cross-subject / mixed / K-trials | Historical protocols in notebooks | `multi_eeg_notebooks/epoch_averaging.ipynb` |
| Time-shifted multi-channel windows | Long epochs → shifted windows | pipeline `SamaraTimeShift` |
| BCI III Pz epoch-average | Subjects A/B validation | pipeline `BCI3PzEpochAverage` |
| Synthetic smoke / CI | Tiny binary tensors | [`configs/synthetic_smoke.json`](configs/synthetic_smoke.json); pipeline `SyntheticBinary` |
| GNN static / edge-learning graphs | Delaunay / k-NN / prior; edge-learn models | `notebooks/` (runner support not in v1) |

Example notebooks: [`notebooks/examples/run_experiment_samara.ipynb`](notebooks/examples/run_experiment_samara.ipynb) (run → load → compare); [`notebooks/examples/samara_speller_compare_colab.ipynb`](notebooks/examples/samara_speller_compare_colab.ipynb) (Colab five-model speller benchmark).

## Speller benchmark

Character-level accuracy and ITR (vs flash repetition count) from a trained binary P300 model. Design: [`docs/superpowers/specs/2026-08-09-bci-speller-benchmark-design.md`](docs/superpowers/specs/2026-08-09-bci-speller-benchmark-design.md).

By default (`use_synthetic: false`) selections come from **real EEG** via binary `data.params` and/or speller `protocol_params`:
- BCI III — `test_mat` + `eloc_path` + `StimulusCode` (ground truth)
- Samara — `path` to `.mat` dir + holdout + 4×4 simulation (`label_source: simulated`)

Set `use_synthetic: true` only for CI smoke (random flashes).

**Samara holdout comes from the binary run, never recomputed.** The speller reads `split_indices.json` from `run_dir` and evaluates only on the epochs that run held out. A missing `split_indices.json` is a hard error unless you opt out with `allow_split_mismatch` / `allow_train_pool_eval`. Provenance is recorded as `eval_source` (`split_indices` | `recomputed` | `train_pool`) in the speller `meta.json`.

Recomputation (the opt-out path) can still diverge if the loader params differ — `channel_idx`, `epoch_len`, `file_pattern`, or the data directory all change which epochs exist, and indices are positional.

```bash
# after a binary run exists under results/<name>_<timestamp>/
python -m pattern_recognition.speller run \
  --config configs/speller_bci3_within.json \
  --run-dir results/<binary_run>/
```

Example configs: [`configs/speller_bci3_within.json`](configs/speller_bci3_within.json) (BCI III row×column), [`configs/speller_samara_sim_within.json`](configs/speller_samara_sim_within.json) (Samara simulated protocol). Artifacts land in `results/<binary_run>/speller/<tag>/`.

## Models

| Model | Supervision | Speller `model_mode` | Notes |
|---|---|---|---|
| `EEGNet` | target / non-target per flash | `flash_scorer` | Binary epoch pipelines; protocol accumulation decode |
| `ContextualTransformer` | target / non-target per flash (contextual) | `flash_scorer` | Selection-packet pipelines; same accumulation as EEGNet |
| `SequenceClassifier` | row+col / cell (protocol heads) | `selection_classifier` | Selection-packet pipelines; direct symbol decode |
| `BaseCNN` | binary epochs | `flash_scorer` | Simple 1D-CNN baseline |
| `DeepConvNet`, `CecottiCNN`, `BaseCNNAttn`, `FlexCNN` | — | — | Importable; not all registered for the runner |
| `SVM` | binary epochs | `flash_scorer` | Registered sklearn RBF baseline; checkpoint `model.joblib` |
| `BaseGNN`, `GIN`, `STGCN`, … | — | — | Static-graph GNN family for 64-ch / SEED notebooks |
| `EdgeLearnGNN`, `PriorEdgeLearnGNN`, … | — | — | Adaptive / learned-edge GNNs |

Runner registers **`SVM`**, **`EEGNet`**, **`BaseCNN`**, **`ContextualTransformer`**, and **`SequenceClassifier`**.

### Training objective

Binary epoch models are trained with `pattern_recognition.losses.BrierLoss` — the Brier score, chosen because it is a proper scoring rule that stays well-behaved under ~1:15 class imbalance. **Binary models emit logits**; the loss applies the softmax itself, so no architecture can silently opt out of being scored properly by forgetting an output activation. The loss reduction averages over samples *and* classes (the standard Brier divided by `n_classes`), matching the scale of the plain `nn.MSELoss()` it replaces so existing learning rates carry over. The sequence models keep their own objectives: masked `BCEWithLogits` with `pos_weight` for `ContextualTransformer`, cross-entropy over the protocol heads for `SequenceClassifier`.

**Three-way P300 comparison** (BCI3 or Samara): train EEGNet on a binary pipeline, CT/SC on `BCI3SelectionPackets` / `SamaraSelectionPackets`, then evaluate with matching speller configs (`flash_scorer` for EEGNet/CT, `selection_classifier` for SC). Stimulus pad index is `0`; BCI3 model codes `1..12` (`num_stimulus_codes=13`), Samara cells `1..16` (`num_stimulus_codes=17`). Samara sequence runs always use `label_source: simulated`. See [`docs/superpowers/specs/2026-08-10-p300-sequence-models-design.md`](docs/superpowers/specs/2026-08-10-p300-sequence-models-design.md).

## Metrics

Reported in run `metrics.json` (and comparison tables):

- **Accuracy** (and CI where used in notebooks)
- **Balanced accuracy** — important under P300 class imbalance
- **F1**
- **Brier score** — multi-category form, `mean_i sum_k (p_ik - o_ik)^2`, range `[0, 2]`, **lower is better**. A proper scoring rule that grades calibrated probabilities instead of the arg-max, so it stays informative under P300 imbalance where accuracy saturates. This is also the training objective (see below)
- **ITR** (Wolpaw, bits/trial) via `pattern_recognition.training.metrics.compute_itr`
- **Device fields** — `device_requested` and `device_resolved` in run artifacts
- **Selection fields** — `best_epoch` and `checkpoint_metric` (which epoch the numbers and the checkpoint come from)

All of the above are validation metrics at `best_epoch`; `history.npz` keeps the full per-epoch curves.

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
# optional GNN / viz stack (historical notebooks):
# poetry install -E all
# or: pip install -e ".[gnn,viz]"
# or: pip install -e .

# smoke (synthetic data, no Samara/BCI files required)
python -m pattern_recognition.experiment run configs/synthetic_smoke.json

# Samara Pz SC N=10 (requires Samara_data/)
python -m pattern_recognition.experiment run configs/samara_pz_eegnet_sc_n10.json
python -m pattern_recognition.experiment run configs/samara_pz_basecnn_sc_n10.json
```

Artifacts land under `results/<name>_<timestamp>/` (`config.json`, `run_meta.json`, `metrics.json`, `history.npz`, `split_indices.json` when the pipeline holds epochs out, optional `model.pt` or `model.joblib` for SVM). `config.json`, `split_indices.json`, and `run_meta.json` are written **before** training starts, so an interrupted run still leaves a readable record; `run_meta.status` is `running` → `completed` or `failed` (with `error`). Run directories are never reused — a second run started in the same second gets a `_2` suffix instead of overwriting the first.

**Best-epoch checkpointing.** `metrics.json` reports the best validation epoch, and `model.pt` holds that epoch's weights — the two always agree. `train.checkpoint_metric` defaults to **`brier`** (lower is better); alternatives are `balanced_accuracy` / `accuracy` / `f1` / `val_loss` / `last`. Ties go to the later epoch, so a metric sitting on a plateau keeps the most-trained weights rather than the first ones. `SequenceClassifier` reports selection accuracy only, so `balanced_accuracy` / `f1` resolve to `accuracy` there — the effective choice is recorded in `run_meta.checkpoint_metric` alongside `best_epoch`. Set `"checkpoint_metric": "last"` for the old final-epoch behaviour. Selecting an epoch on validation data makes those validation metrics mildly optimistic; the speller benchmark is unaffected because it scores a holdout training never sees.

Brier is the default because it keeps ranking epochs while a P300 model still predicts all-negative: accuracy then sits at the majority rate and balanced accuracy at exactly 0.5. On a real Samara subject, balanced accuracy was `0.5000` for every epoch while Brier went `0.1333 → 0.1179 → 0.1163 → 0.1166 → 0.1153`.

Configs reject unknown keys (`extra="forbid"`): a misspelled or misplaced field fails validation instead of silently changing the experiment. Set the split seed in exactly one place — `split.seed` and `data.params.seed` conflicting is an error, not a silent override.

Programmatic:

```python
from pattern_recognition.experiment import run_experiment
from pattern_recognition.reporting import load_run, metrics_table, compare_runs

run_dir = run_experiment("configs/synthetic_smoke.json")
run = load_run(run_dir)
print(run.metrics)
```

### Colab

Colab uses **Python 3.13**. Core install accepts Colab’s Torch and NumPy 2.x — one command:

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content/drive/MyDrive/pattern_recognition
!pip install -q -e .
# or: !pip install -q "git+https://github.com/Sidl419/pattern_recognition.git"

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

Point `data.params.path` / `output_dir` at Drive mounts. Use `"device": "cuda"` on GPU runtimes (hard-fails if no GPU); `"auto"` is fine too. See [`notebooks/examples/samara_speller_compare_colab.ipynb`](notebooks/examples/samara_speller_compare_colab.ipynb) for the five-model compare flow. GNN notebooks need `pip install -e ".[gnn,viz]"`.

## Repo map

```text
pattern_recognition/
  selections/          # shared paradigm layer: grids, protocols, packing, decode
  data/                # loaders, splits, pipelines (registry-driven)
  models/              # CNN / GNN / sklearn model registry
  training/            # loops, metrics, best-epoch checkpointing, device
  experiment/          # config schema + run_experiment + CLI
  speller/             # character-level benchmark (depends on all of the above)
  reporting/           # read run artifacts -> tables and plots
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
make test
# or: poetry run pytest tests/ -v
make format        # Ruff format
make check         # format-check + lint + tests
```

GitHub Actions runs the same suite on push/PR to `main` (CPU Torch; `torch-geometric` is best-effort so CNN/unit tests still run if PyG wheels fail).
