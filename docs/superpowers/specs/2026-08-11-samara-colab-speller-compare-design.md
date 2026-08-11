# Samara Colab speller comparison (SVM + CNNs + sequence)

**Date:** 2026-08-11  
**Status:** Approved / implemented  
**Scope:** Register sklearn `SVM` in the experiment runner as a first-class `flash_scorer`; add Samara `n_average=1` train/speller configs; ship a Colab notebook that trains **SVM, EEGNet, BaseCNN, ContextualTransformer, SequenceClassifier** and compares them on the Samara within-subject character (speller) benchmark.

**Related:**

- `docs/superpowers/specs/2026-08-08-package-and-experiment-runner-design.md`
- `docs/superpowers/specs/2026-08-09-bci-speller-benchmark-design.md`
- `docs/superpowers/specs/2026-08-10-p300-sequence-models-design.md`

## Goals

1. End-to-end Colab workflow: install → mount Samara data → train five models → run speller → one comparison table / `char_acc(r)` plot.
2. Treat SVM like other flash scorers: `run_experiment(config)` → `results/<run>/` → `run_speller_benchmark` without a notebook-only `scores_provider` path (provider remains available for tests).
3. Fair flash-scorer comparison on Samara: **within-subject**, binary epochs with **`n_average=1`**, shared `split` (`seed=0`, `epoch_holdout=0.3`, `stratify=true`, `val_fraction=0.2`), phrase `JUST_DO_IT`, repetitions `[1, 2, 5, 10]`.
4. Research-scale neural training defaults in the notebook/configs: **`num_epochs=250`** (cell override to 400/500), matching historical notebooks rather than the thin `50` in older example JSONs.

## Non-goals

- Cross-subject / LOSO in this notebook (toggle later if needed).
- Dual `n_average=1` and `n_average=10` curves in v1 of the notebook.
- Changing EEGNet / BaseCNN / CT / SC architecture math.
- Hyperparameter search, multi-GPU DDP, or GNN.
- Committing Samara `.mat` / large CSV dumps.
- Replacing existing `*_n10` configs (keep them; add parallel `*_n1` configs).

## Locked decisions

| Topic | Choice |
|---|---|
| SVM → speller | Full runner registration + joblib checkpoint + `SklearnFlashScorer` |
| Subject mode | Within-subject only |
| Binary averaging | `n_average=1` |
| Epoch budget | Default 250 for neural models; notebook `NUM_EPOCHS` override |
| SVM “epochs” | One-shot `SVC.fit` (ignore `num_epochs` for training steps; may still appear in config for schema uniformity) |
| Comparison metric | Speller `char_acc(r)` (+ ITR curves from existing artifacts); optional binary `metrics.json` side table |

## Architecture

```text
Train
  SVM / EEGNet / BaseCNN     SamaraWithinSubjectAverage(n_average=1)  → flash_scorer
  ContextualTransformer      SamaraSelectionPackets                   → flash_scorer
  SequenceClassifier         SamaraSelectionPackets                   → selection_classifier

Artifacts (run_dir)
  config.json, run_meta.json (model_mode), metrics.json, history.npz,
  split_indices.json (when pipeline emits them),
  model.pt          # Torch models
  model.joblib      # SVM only

Eval
  flash_scorer + Torch     → RunFlashScorer / ContextualFlashScorer
  flash_scorer + SVM       → SklearnFlashScorer (model.joblib)
  selection_classifier     → RunSelectionClassifier
```

### Comparison table (notebook target)

| Model | Pipeline | `model_mode` | Decode |
|---|---|---|---|
| SVM | `SamaraWithinSubjectAverage` `n_average=1` | `flash_scorer` | protocol accumulation |
| EEGNet | same | `flash_scorer` | same |
| BaseCNN | same | `flash_scorer` | same |
| ContextualTransformer | `SamaraSelectionPackets` | `flash_scorer` | same |
| SequenceClassifier | `SamaraSelectionPackets` | `selection_classifier` | direct symbol |

## SVM registration (Approach 1)

### Factory

- `@register_model("SVM")` returns a small non-`nn.Module` holder (e.g. `SklearnSVM`) storing constructor kwargs (`C`, `kernel`, `gamma`, `probability`, …) and, after fit, the fitted `sklearn.svm.SVC`.
- Default params (overridable in config): `C=1.0`, `kernel="rbf"`, `probability=True` (prefer `predict_proba[:, 1]` for scores; fall back to `decision_function` if `probability=False`).

### Runner branch

- Add `CLASSICAL_MODELS = frozenset({"SVM"})` alongside `SEQUENCE_MODELS`.
- Forbidden: selection-packet pipelines for SVM (same rule as EEGNet/BaseCNN).
- Flow:
  1. Build binary pipeline bundle as today.
  2. Materialize train/val arrays from datasets (flatten channels×time → feature vector per epoch).
  3. `train_svm(...)` → fit on train; evaluate binary metrics on val (accuracy, balanced accuracy, F1; ITR via existing helper if applicable, else `nan`).
  4. History arrays length **1** (single eval after fit) so reporting still loads `history.npz`.
  5. `run_meta.model_mode = "flash_scorer"`.
  6. If `save_model`: `joblib.dump(fitted_svc, run_dir / "model.joblib")` — **do not** write `model.pt`.

### Speller loading

- Extend checkpoint resolution:
  - If `config.model.name == "SVM"` (or `model.joblib` present and no `model.pt`): load joblib → `SklearnFlashScorer`.
  - Else existing Torch path.
- `SklearnFlashScorer.predict_scores(selection)`: reshape flashes to `(n_flashes, -1)`, score with proba/decision_function, return `float32` vector length `n_flashes`.
- Fail clearly if neither `model.pt` nor `model.joblib` exists.

### Dependencies

- `scikit-learn` and `joblib` already used in pipelines; ensure Poetry deps cover joblib explicitly if not transitive-only.

## Configs

Add (paths under `configs/`):

| File | Role |
|---|---|
| `samara_pz_svm_sc_n1.json` | SVM + `n_average=1` + shared `split` |
| `samara_pz_eegnet_sc_n1.json` | EEGNet + `n_average=1` + shared `split`, `num_epochs=250`, `save_model=true` |
| `samara_pz_basecnn_sc_n1.json` | BaseCNN analogue |
| `speller_samara_flash_n1.json` | Template flash_scorer speller (`run_dir` placeholder); used for SVM/EEGNet/BaseCNN/CT |
| Keep | `samara_contextual_transformer.json`, `samara_sequence_classifier.json`, `speller_samara_contextual.json`, `speller_samara_sequence_classifier.json` — bump `num_epochs` to 250 in train configs (or notebook overrides in-memory) |

All Samara paths remain `Samara_data/` relative to repo root; Colab sets cwd / env so that resolves (or notebook rewrites `path` after mount).

Shared train `split` block for binary + sequence Samara configs used by this notebook:

```json
"split": {
  "seed": 0,
  "epoch_holdout": 0.3,
  "stratify": true,
  "val_fraction": 0.2
}
```

Speller templates: `subject_mode: within_subject`, `protocol: samara_single_flash_sim`, `repetitions: [1,2,5,10]`, same `split` / `simulation.phrase`, `plots: true`.

## Colab notebook

**Path:** `notebooks/examples/samara_speller_compare_colab.ipynb`

### Sections

1. **Title / purpose** — five-model Samara within-subject character benchmark.
2. **Colab setup** — `pip install` from GitHub or `-e .` after Drive clone; mount Drive; set `REPO_ROOT`, `SAMARA_PATH`, `NUM_EPOCHS=250`, `DEVICE=auto`.
3. **Sanity** — assert `.mat` files exist; print subject count.
4. **Train loop** — for each model config (dict or JSON with path rewrite): `run_experiment` → collect `run_dirs[name]`.
5. **Speller loop** — for each run: build speller config with correct `model_mode` and `run_dir`; `run_speller_benchmark`.
6. **Compare** — load each `speller/<tag>/` metrics; table of `char_acc` at r∈{1,2,5,10}; overlay plot; optional binary `metrics_table` for flash models.
7. **Local note** — same cells work locally if `Samara_data/` is present; Colab cells are optional.

### Notebook policy

- Absolute imports only (`from pattern_recognition...`).
- Do not clear outputs unless asked.
- Prefer rewriting config dicts in cells over committing Colab-specific paths.
- Heavy training is expected; no synthetic fallback required for this notebook (fail clearly if data missing). Optional tiny smoke cell may use SyntheticBinary/SyntheticSelectionPackets for CI documentation only — **not** the main path.

## Tests

- Unit: `SklearnFlashScorer` shapes; SVM factory registers; runner writes `model.joblib` + `model_mode` on a tiny SyntheticBinary config; speller smoke with SVM joblib on synthetic/oracle-scale data.
- Parametrize existing split-config pairing tests to include new `*_n1` + speller templates where applicable.
- Do not require real Samara `.mat` in CI.

## Documentation

- README: mention SVM as classical flash scorer; link notebook; note `model.joblib` artifact.
- Do not expand historical multi_eeg notebooks.

## Implementation order

1. SVM model + `train_svm` + runner branch + joblib save.
2. Speller `SklearnFlashScorer` + checkpoint resolve.
3. Configs (`*_n1`, speller template; epoch bumps for CT/SC used by notebook).
4. Tests.
5. Colab notebook + README blurb.
6. `make check`.

## Open points (resolved in this spec)

- SVM adapter location: **package**, not notebook-only.
- Epoch default: **250** with notebook override (user historical range 100–500).
- Averaging: **n_average=1** only for this comparison.
