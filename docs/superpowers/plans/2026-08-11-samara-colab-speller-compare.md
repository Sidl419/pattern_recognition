# Samara Colab Speller Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register sklearn `SVM` as a first-class flash scorer in the experiment runner, add Samara `n_average=1` configs, and ship a Colab notebook that trains SVM / EEGNet / BaseCNN / ContextualTransformer / SequenceClassifier and compares them on the Samara within-subject speller benchmark.

**Architecture:** Classical branch in `run_experiment` fits `SVC`, writes `model.joblib` + standard run artifacts (`model_mode=flash_scorer`). Speller loads via `SklearnFlashScorer`. Notebook drives five configs → five speller runs → `char_acc(r)` table/plot. Spec: `docs/superpowers/specs/2026-08-11-samara-colab-speller-compare-design.md`.

**Tech Stack:** Python, scikit-learn (`SVC`), joblib, PyTorch (existing neural path), Pydantic configs, pytest, Jupyter.

## Global Constraints

- Absolute imports only: `from pattern_recognition...`
- Speller modes stay exactly `flash_scorer` | `selection_classifier` (no third mode)
- Samara comparison: within-subject, `n_average=1`, shared split `seed=0` / `epoch_holdout=0.3` / `stratify=true` / `val_fraction=0.2`, phrase `JUST_DO_IT`, repetitions `[1, 2, 5, 10]`
- Neural default `num_epochs=250` (notebook may override); SVM is one-shot fit
- Do not change EEGNet / BaseCNN / CT / SC architecture math
- Do not replace existing `*_n10` configs; add parallel `*_n1`
- Do not commit `Samara_data/`, `raw_data/`, `processed_data/`, or large `.mat`/CSV
- CI must not require real Samara dumps (SyntheticBinary + synthetic speller only)

---

## File structure

| File | Responsibility |
|---|---|
| `pattern_recognition/models/classical.py` | `SklearnSVM` holder + fit/score helpers |
| `pattern_recognition/models/__init__.py` | `@register_model("SVM")` |
| `pattern_recognition/training/svm_loop.py` | Materialize XY, fit, binary val metrics, history length 1 |
| `pattern_recognition/experiment/runner.py` | `CLASSICAL_MODELS` branch; save `model.joblib` |
| `pattern_recognition/speller/benchmark.py` | `SklearnFlashScorer`; resolve joblib vs `model.pt` |
| `pattern_recognition/speller/__init__.py` | Export `SklearnFlashScorer` if useful |
| `pyproject.toml` | Explicit `joblib` dependency |
| `configs/samara_pz_svm_sc_n1.json` | SVM train |
| `configs/samara_pz_eegnet_sc_n1.json` | EEGNet n1 + split + 250 epochs |
| `configs/samara_pz_basecnn_sc_n1.json` | BaseCNN n1 |
| `configs/speller_samara_flash_n1.json` | Flash speller template |
| `configs/samara_contextual_transformer.json` | Bump `num_epochs` → 250 |
| `configs/samara_sequence_classifier.json` | Bump `num_epochs` → 250 |
| `notebooks/examples/samara_speller_compare_colab.ipynb` | Colab / local five-way compare |
| `README.md` | SVM row + notebook link + `model.joblib` |
| `tests/test_svm_model.py` | Factory + SklearnSVM score shapes |
| `tests/test_svm_run_artifacts.py` | Runner joblib contract |
| `tests/speller/test_sklearn_scorer.py` | Scorer + load from SVM run |
| `tests/test_registries.py` | Expect `SVM` in `list_models()` |
| `tests/test_split_config.py` | Parametrize new n1 config pairs |

---

### Task 1: SklearnSVM model + registry

**Files:**
- Create: `pattern_recognition/models/classical.py`
- Modify: `pattern_recognition/models/__init__.py`
- Modify: `tests/test_registries.py`
- Create: `tests/test_svm_model.py`

**Interfaces:**
- Produces: `class SklearnSVM` with `__init__(**svc_kwargs)`, `fit(X, y) -> Self`, `predict_scores(X) -> np.ndarray`, `fitted_estimator` property; `build_svm(**params) -> SklearnSVM` registered as `"SVM"`
- Consumes: `sklearn.svm.SVC`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_svm_model.py
import numpy as np
import pattern_recognition.models  # noqa: F401
from pattern_recognition.models import get_model
from pattern_recognition.models.classical import SklearnSVM


def test_svm_registered():
    assert "SVM" in __import__(
        "pattern_recognition.models", fromlist=["list_models"]
    ).list_models()


def test_sklearn_svm_fit_and_scores():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 32)).astype(np.float64)
    y = rng.integers(0, 2, size=40)
    model = SklearnSVM(C=1.0, kernel="linear", probability=True)
    model.fit(X, y)
    scores = model.predict_scores(X[:5])
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))


def test_build_svm_via_registry():
    clf = get_model("SVM")(C=0.5, kernel="rbf", probability=True)
    assert isinstance(clf, SklearnSVM)
```

Also update `tests/test_registries.py` expected model list to include `"SVM"` wherever names are asserted.

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_svm_model.py -v`

Expected: FAIL (import / `SVM` not registered)

- [ ] **Step 3: Implement SklearnSVM + register**

```python
# pattern_recognition/models/classical.py
from __future__ import annotations

from typing import Any, Self

import numpy as np
from sklearn.svm import SVC


class SklearnSVM:
    """Thin holder for sklearn ``SVC`` used as a binary flash scorer."""

    def __init__(self, **svc_kwargs: Any) -> None:
        params = {
            "C": 1.0,
            "kernel": "rbf",
            "probability": True,
            **svc_kwargs,
        }
        self.svc_kwargs = params
        self._clf: SVC | None = None

    @property
    def fitted_estimator(self) -> SVC:
        if self._clf is None:
            raise RuntimeError("SklearnSVM is not fitted")
        return self._clf

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        clf = SVC(**self.svc_kwargs)
        clf.fit(X, y)
        self._clf = clf
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        clf = self.fitted_estimator
        X = np.asarray(X, dtype=np.float64)
        if getattr(clf, "probability", False):
            return clf.predict_proba(X)[:, 1].astype(np.float32)
        return clf.decision_function(X).astype(np.float32)
```

```python
# pattern_recognition/models/__init__.py — add:
@register_model("SVM")
def build_svm(**params: Any):
    from pattern_recognition.models.classical import SklearnSVM

    return SklearnSVM(**params)
```

Export `build_svm` in `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_svm_model.py tests/test_registries.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/models/classical.py pattern_recognition/models/__init__.py tests/test_svm_model.py tests/test_registries.py
git commit -m "$(cat <<'EOF'
feat: register SklearnSVM as runner model SVM

EOF
)"
```

---

### Task 2: `train_svm` + classical runner branch

**Files:**
- Create: `pattern_recognition/training/svm_loop.py`
- Modify: `pattern_recognition/experiment/runner.py`
- Modify: `pyproject.toml` (add `joblib = "^1.3.0"` or current compatible floor under `[tool.poetry.dependencies]`)
- Create: `tests/test_svm_run_artifacts.py`

**Interfaces:**
- Consumes: `SklearnSVM.fit` / `predict_scores`; `CNNMatrixDataset.tensors`; `compute_itr`
- Produces: `train_svm(model, train_ds, val_ds) -> tuple[val_loss_history, acc_dict, time_elapsed]` with history length 1; runner writes `model.joblib` (not `model.pt`) and `run_meta.model_mode == "flash_scorer"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_svm_run_artifacts.py
import json
from pathlib import Path

import joblib
import pattern_recognition.models  # noqa: F401
from pattern_recognition.experiment import run_experiment


def _svm_cfg(tmp_path: Path) -> dict:
    return {
        "name": "svm_smoke",
        "seed": 0,
        "device": "cpu",
        "data": {
            "pipeline": "SyntheticBinary",
            "params": {
                "n_train": 32,
                "n_val": 16,
                "n_channels": 1,
                "n_times": 64,
            },
        },
        "model": {
            "name": "SVM",
            "params": {"C": 1.0, "kernel": "linear", "probability": True},
        },
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 8,
            "num_epochs": 1,
            "step_size": 1,
            "gamma": 1.0,
            "save_model": True,
        },
        "output_dir": str(tmp_path),
    }


def test_svm_run_writes_joblib_not_pt(tmp_path: Path):
    run_dir = run_experiment(_svm_cfg(tmp_path))
    assert (run_dir / "model.joblib").is_file()
    assert not (run_dir / "model.pt").exists()
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["model_mode"] == "flash_scorer"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    for key in ("accuracy", "balanced_accuracy", "f1", "itr", "train_time_sec"):
        assert key in metrics
    hist = __import__("numpy").load(run_dir / "history.npz")
    assert hist["accuracy"].shape == (1,)
    clf = joblib.load(run_dir / "model.joblib")
    assert hasattr(clf, "predict")


def test_svm_rejects_selection_packets(tmp_path: Path):
    import pytest

    cfg = _svm_cfg(tmp_path)
    cfg["data"] = {
        "pipeline": "SyntheticSelectionPackets",
        "params": {"n_train": 4, "n_val": 2, "n_channels": 1, "n_times": 64},
    }
    with pytest.raises(ValueError, match="SelectionPackets"):
        run_experiment(cfg)
```

Adjust `SyntheticSelectionPackets` params to match the real pipeline constructor if names differ — read `pattern_recognition/data/pipelines/synthetic_selection.py` before implementing the negative test.

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_svm_run_artifacts.py -v`

Expected: FAIL (classical branch missing / still tries `nn.Module` train)

- [ ] **Step 3: Implement `train_svm` + runner branch**

```python
# pattern_recognition/training/svm_loop.py
from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from pattern_recognition.models.classical import SklearnSVM
from pattern_recognition.training.metrics import compute_itr


def dataset_to_xy(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ``CNNMatrixDataset`` tensors to ``(n, feat)`` and int labels."""
    if not hasattr(dataset, "tensors"):
        raise TypeError(
            "SVM training requires a tensor dataset with .tensors "
            f"(got {type(dataset)!r})"
        )
    x = dataset.tensors[0].detach().cpu().numpy()
    y = dataset.tensors[1].detach().cpu().numpy().astype(int).reshape(-1)
    x = x.reshape(x.shape[0], -1)
    return x, y


def train_svm(
    model: SklearnSVM,
    train_ds: Any,
    val_ds: Any,
) -> tuple[list[float], dict[str, np.ndarray], float]:
    started = time.time()
    x_tr, y_tr = dataset_to_xy(train_ds)
    x_va, y_va = dataset_to_xy(val_ds)
    model.fit(x_tr, y_tr)
    scores = model.predict_scores(x_va)
    # threshold 0.5 for proba; for decision_function use sign
    if model.svc_kwargs.get("probability", False):
        y_hat = (scores >= 0.5).astype(int)
    else:
        y_hat = (scores >= 0.0).astype(int)
    acc = float(accuracy_score(y_va, y_hat))
    bacc = float(balanced_accuracy_score(y_va, y_hat))
    f1 = float(f1_score(y_va, y_hat, zero_division=0))
    itr = float(compute_itr(acc, n_classes=2))
    elapsed = float(time.time() - started)
    # Match runner history keys used by train_model
    acc_dict = {
        "Accuracy": np.array([acc], dtype=float),
        "Balanced Accuracy": np.array([bacc], dtype=float),
        "F1-score": np.array([f1], dtype=float),
        "ITR": np.array([itr], dtype=float),
    }
    val_loss_history = [float("nan")]
    return val_loss_history, acc_dict, elapsed
```

In `runner.py`:

```python
CLASSICAL_MODELS = frozenset({"SVM"})
BINARY_PACKET_FORBIDDEN = frozenset({"EEGNet", "BaseCNN", "SVM"})
```

Branch before Torch train:

```python
is_classical = cfg.model.name in CLASSICAL_MODELS
# ... existing sequence / packet guards (SVM already in BINARY_PACKET_FORBIDDEN)

model = get_model(cfg.model.name)(**cfg.model.params)

if is_classical:
    from pattern_recognition.training.svm_loop import train_svm

    val_loss_history, acc_dict, time_elapsed = train_svm(
        model, bundle.train, bundle.val
    )
    model_mode = "flash_scorer"
elif is_sequence:
    ...
else:
    ...
```

Save path:

```python
if cfg.train.save_model:
    if is_classical:
        import joblib

        joblib.dump(model.fitted_estimator, run_dir / "model.joblib")
    else:
        torch.save(model.state_dict(), run_dir / "model.pt")
```

Add `joblib` to `pyproject.toml` dependencies and run `poetry lock` / `poetry install` as needed.

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_svm_run_artifacts.py tests/test_run_artifacts.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/training/svm_loop.py pattern_recognition/experiment/runner.py pyproject.toml poetry.lock tests/test_svm_run_artifacts.py
git commit -m "$(cat <<'EOF'
feat: train SVM via classical run_experiment branch

EOF
)"
```

---

### Task 3: SklearnFlashScorer + checkpoint resolve

**Files:**
- Modify: `pattern_recognition/speller/benchmark.py`
- Modify: `pattern_recognition/speller/__init__.py` (export `SklearnFlashScorer`)
- Create: `tests/speller/test_sklearn_scorer.py`
- Modify: `tests/speller/test_load_flash_scorer.py` (missing-artifact message may change)

**Interfaces:**
- Consumes: joblib `SVC`; `Selection.flashes`
- Produces: `class SklearnFlashScorer` with `predict_scores(selection) -> np.ndarray`; `load_flash_scorer_from_run` returns it for SVM runs

- [ ] **Step 1: Write the failing tests**

```python
# tests/speller/test_sklearn_scorer.py
from pathlib import Path

import numpy as np
import pattern_recognition.models  # noqa: F401
from pattern_recognition.experiment import run_experiment
from pattern_recognition.speller.benchmark import (
    SklearnFlashScorer,
    load_flash_scorer_from_run,
    run_speller_benchmark,
)
from pattern_recognition.speller.types import Selection


def _svm_run(tmp_path: Path) -> Path:
    return run_experiment(
        {
            "name": "svm_speller",
            "seed": 0,
            "device": "cpu",
            "data": {
                "pipeline": "SyntheticBinary",
                "params": {
                    "n_train": 32,
                    "n_val": 16,
                    "n_channels": 1,
                    "n_times": 64,
                },
            },
            "model": {
                "name": "SVM",
                "params": {"C": 1.0, "kernel": "linear", "probability": True},
            },
            "train": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "batch_size": 8,
                "num_epochs": 1,
                "step_size": 1,
                "gamma": 1.0,
                "save_model": True,
            },
            "output_dir": str(tmp_path),
        }
    )


def test_sklearn_flash_scorer_shapes():
    from sklearn.svm import SVC

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 64))
    y = rng.integers(0, 2, size=20)
    clf = SVC(kernel="linear", probability=True).fit(X, y)
    scorer = SklearnFlashScorer(clf)
    flashes = rng.normal(size=(5, 1, 64)).astype(np.float32)
    sel = Selection(
        flashes=flashes,
        stimulus_ids=np.arange(5),
        target_char="A",
        repeat_index=np.zeros(5, dtype=np.int64),
        meta={},
    )
    scores = scorer.predict_scores(sel)
    assert scores.shape == (5,)
    assert scores.dtype == np.float32


def test_load_flash_scorer_from_svm_run(tmp_path: Path):
    run_dir = _svm_run(tmp_path)
    scorer = load_flash_scorer_from_run(run_dir, model_mode="flash_scorer")
    assert isinstance(scorer, SklearnFlashScorer)
    flashes = np.random.randn(4, 1, 64).astype(np.float32)
    sel = Selection(
        flashes=flashes,
        stimulus_ids=np.arange(4),
        target_char="A",
        repeat_index=np.zeros(4, dtype=np.int64),
        meta={},
    )
    scores = scorer.predict_scores(sel)
    assert scores.shape == (4,)


def test_speller_smoke_with_svm_run(tmp_path: Path):
    run_dir = _svm_run(tmp_path)
    out = run_speller_benchmark(
        {
            "tag": "svm_syn",
            "model_mode": "flash_scorer",
            "protocol": "bci3_rowcol",
            "subject_mode": "within_subject",
            "repetitions": [1, 2],
            "run_dir": str(run_dir),
            "use_synthetic": True,
            "plots": False,
        }
    )
    assert Path(out).is_dir()
```

If synthetic speller config needs more fields, copy the minimal working payload from `tests/speller/test_benchmark_smoke.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/speller/test_sklearn_scorer.py -v`

Expected: FAIL (`SklearnFlashScorer` missing / load requires `model.pt`)

- [ ] **Step 3: Implement scorer + loader**

```python
# in benchmark.py
class SklearnFlashScorer:
    """Score flashes with a fitted sklearn classifier (e.g. SVC)."""

    def __init__(self, estimator) -> None:
        self._estimator = estimator

    def predict_scores(self, selection: Selection) -> np.ndarray:
        flashes = np.asarray(selection.flashes, dtype=np.float64)
        if flashes.ndim == 1:
            flashes = flashes[np.newaxis, ...]
        x = flashes.reshape(flashes.shape[0], -1)
        est = self._estimator
        if getattr(est, "probability", False) and hasattr(est, "predict_proba"):
            scores = est.predict_proba(x)[:, 1]
        else:
            scores = est.decision_function(x)
        return np.asarray(scores, dtype=np.float32)
```

Refactor `_load_run_checkpoint` / `load_flash_scorer_from_run`:

1. Load `config.json` always.
2. If `exp_cfg.model.name == "SVM"`:
   - Require `model.joblib`
   - `clf = joblib.load(...)`
   - return `SklearnFlashScorer(clf)` (no Torch model construction)
3. Else existing Torch `model.pt` path.
4. Error if neither artifact exists: message mentioning both `model.pt` and `model.joblib`.

Update `test_load_flash_scorer_missing_artifacts` if the regex changes.

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/speller/test_sklearn_scorer.py tests/speller/test_load_flash_scorer.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pattern_recognition/speller/benchmark.py pattern_recognition/speller/__init__.py tests/speller/test_sklearn_scorer.py tests/speller/test_load_flash_scorer.py
git commit -m "$(cat <<'EOF'
feat: load SVM flash scores via SklearnFlashScorer

EOF
)"
```

---

### Task 4: Samara n1 configs + epoch bumps + split pairing tests

**Files:**
- Create: `configs/samara_pz_svm_sc_n1.json`
- Create: `configs/samara_pz_eegnet_sc_n1.json`
- Create: `configs/samara_pz_basecnn_sc_n1.json`
- Create: `configs/speller_samara_flash_n1.json`
- Modify: `configs/samara_contextual_transformer.json` (`num_epochs`: 250)
- Modify: `configs/samara_sequence_classifier.json` (`num_epochs`: 250)
- Modify: `tests/test_split_config.py`

**Interfaces:**
- Produces: validated JSON configs with shared top-level `split` matching speller templates

- [ ] **Step 1: Write failing split-pairing assertions**

Extend `test_samara_train_example_configs_have_top_level_split` parametrize with:

```python
("samara_pz_eegnet_sc_n1.json", "speller_samara_flash_n1.json"),
("samara_pz_basecnn_sc_n1.json", "speller_samara_flash_n1.json"),
("samara_pz_svm_sc_n1.json", "speller_samara_flash_n1.json"),
```

Keep existing CT/SC pairs.

Add a small test that EEGNet/BaseCNN/SVM n1 configs have `n_average == 1` and `num_epochs == 250` (SVM may keep `num_epochs` for schema only).

- [ ] **Step 2: Run to verify fail**

Run: `poetry run pytest tests/test_split_config.py::test_samara_train_example_configs_have_top_level_split -v`

Expected: FAIL (files missing)

- [ ] **Step 3: Write configs**

`configs/samara_pz_svm_sc_n1.json`:

```json
{
  "name": "samara_pz_svm_sc_n1",
  "seed": 0,
  "device": "auto",
  "split": {
    "seed": 0,
    "epoch_holdout": 0.3,
    "stratify": true,
    "val_fraction": 0.2
  },
  "data": {
    "pipeline": "SamaraWithinSubjectAverage",
    "params": {
      "path": "Samara_data/",
      "channel_idx": 1,
      "n_average": 1,
      "mode": "SC"
    }
  },
  "model": {
    "name": "SVM",
    "params": {
      "C": 1.0,
      "kernel": "rbf",
      "probability": true
    }
  },
  "train": {
    "lr": 0.0001,
    "weight_decay": 0.01,
    "batch_size": 64,
    "num_epochs": 1,
    "step_size": 1,
    "gamma": 1.0,
    "save_model": true
  },
  "output_dir": "results/"
}
```

EEGNet / BaseCNN n1: same `split` + `n_average: 1`, `num_epochs: 250`, model params as n10 configs, `seed: 0` to match split.

`configs/speller_samara_flash_n1.json`: copy `speller_samara_sim_within.json` structure with `"tag": "samara_flash_n1_r10"`, same split, `run_dir` placeholder `"results/<binary_or_ct_run>/"`.

Bump CT/SC train `num_epochs` to `250`.

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_split_config.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/samara_pz_svm_sc_n1.json configs/samara_pz_eegnet_sc_n1.json configs/samara_pz_basecnn_sc_n1.json configs/speller_samara_flash_n1.json configs/samara_contextual_transformer.json configs/samara_sequence_classifier.json tests/test_split_config.py
git commit -m "$(cat <<'EOF'
chore: add Samara n1 train/speller configs for five-way compare

EOF
)"
```

---

### Task 5: Colab notebook + README

**Files:**
- Create: `notebooks/examples/samara_speller_compare_colab.ipynb`
- Modify: `README.md`

**Interfaces:**
- Consumes: `run_experiment`, `run_speller_benchmark`, reporting helpers, configs from Task 4

- [ ] **Step 1: Create notebook cells** (nbformat; no need to execute training in CI)

Notebook structure (markdown + code):

1. Title — five-model Samara within-subject character benchmark; link design spec.
2. Colab install / Drive mount (optional cells).
3. Paths:

```python
from pathlib import Path
import json
import copy

REPO_ROOT = Path(".").resolve()
if not (REPO_ROOT / "configs").is_dir() and (REPO_ROOT.parent.parent / "configs").is_dir():
    REPO_ROOT = REPO_ROOT.parent.parent

SAMARA_PATH = REPO_ROOT / "Samara_data"  # override on Colab, e.g. Drive path
NUM_EPOCHS = 250
DEVICE = "auto"
OUTPUT_DIR = REPO_ROOT / "results"
```

4. Sanity: `assert SAMARA_PATH.is_dir() and any(SAMARA_PATH.glob("*.mat"))`.
5. Load configs as dicts; rewrite `data.params.path` / `protocol_params.path` to `str(SAMARA_PATH)`; set `train.num_epochs = NUM_EPOCHS` for neural models; set `device` / `output_dir`.
6. Train map:

```python
from pattern_recognition.experiment import run_experiment

TRAIN_CONFIGS = {
    "SVM": "samara_pz_svm_sc_n1.json",
    "EEGNet": "samara_pz_eegnet_sc_n1.json",
    "BaseCNN": "samara_pz_basecnn_sc_n1.json",
    "ContextualTransformer": "samara_contextual_transformer.json",
    "SequenceClassifier": "samara_sequence_classifier.json",
}
run_dirs = {}
for name, fname in TRAIN_CONFIGS.items():
    cfg = json.loads((REPO_ROOT / "configs" / fname).read_text())
    # path / epoch / device rewrites here
    run_dirs[name] = run_experiment(cfg)
```

7. Speller:

```python
from pattern_recognition.speller import run_speller_benchmark

SPELLER_MODE = {
    "SVM": ("flash_scorer", "speller_samara_flash_n1.json"),
    "EEGNet": ("flash_scorer", "speller_samara_flash_n1.json"),
    "BaseCNN": ("flash_scorer", "speller_samara_flash_n1.json"),
    "ContextualTransformer": ("flash_scorer", "speller_samara_contextual.json"),
    "SequenceClassifier": ("selection_classifier", "speller_samara_sequence_classifier.json"),
}
speller_dirs = {}
for name, run_dir in run_dirs.items():
    mode, fname = SPELLER_MODE[name]
    cfg = json.loads((REPO_ROOT / "configs" / fname).read_text())
    cfg["run_dir"] = str(run_dir)
    cfg["model_mode"] = mode
    cfg["tag"] = f"compare_{name.lower()}"
    # rewrite protocol_params.path
    speller_dirs[name] = run_speller_benchmark(cfg)
```

8. Compare: load each `metrics.json` under speller dirs; build pandas table of `char_acc` vs `r`; matplotlib overlay; optional `metrics_table(list(run_dirs.values()))` for binary train metrics.

Absolute imports only. Fail clearly if data missing (no synthetic main path).

- [ ] **Step 2: Update README**

- Models table: SVM row → registered `SVM`, `flash_scorer`, `model.joblib`.
- Runner registers list includes `SVM`.
- Example notebook link to `notebooks/examples/samara_speller_compare_colab.ipynb`.
- Artifacts line mentions optional `model.joblib`.
- Experiment setups: mention `*_sc_n1` configs for flash-level fair compare.

- [ ] **Step 3: Smoke-check notebook JSON validity**

Run: `poetry run python -c "import nbformat; nbformat.read('notebooks/examples/samara_speller_compare_colab.ipynb', as_version=4); print('ok')"`

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add notebooks/examples/samara_speller_compare_colab.ipynb README.md
git commit -m "$(cat <<'EOF'
docs: add Samara Colab five-model speller compare notebook

EOF
)"
```

---

### Task 6: Full verification

**Files:** none new

- [ ] **Step 1: Format + lint + test**

Run: `make check`

Expected: format-check / lint clean; all tests PASS (including new SVM / speller / split tests).

- [ ] **Step 2: Fix any failures**

Address Ruff or pytest issues without changing model math for neural nets.

- [ ] **Step 3: Commit if fixes needed**

```bash
git add -u
git commit -m "$(cat <<'EOF'
fix: address check failures for SVM speller compare

EOF
)"
```

- [ ] **Step 4: Mark design implemented**

Set spec status line to `Approved / implemented` if all tasks done.

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| Register SVM + classical train + `model.joblib` | 1–2 |
| `SklearnFlashScorer` + load from run | 3 |
| Within-subject, `n_average=1`, shared split, phrase, r grid | 4–5 |
| `num_epochs=250` + notebook override | 4–5 |
| Five-model Colab notebook + compare | 5 |
| Tests without real Samara | 1–3 |
| README | 5 |
| `make check` | 6 |
| Keep `*_n10` | 4 (add only) |

## Placeholder / consistency check

- Scorer API: `predict_scores(selection) -> np.ndarray` (matches `FlashScorer`)
- Artifact: always `model.joblib` for SVM; never `model.pt`
- Forbidden packet pipelines include `"SVM"` with EEGNet/BaseCNN
- History keys: `"Accuracy"`, `"Balanced Accuracy"`, `"F1-score"`, `"ITR"` (runner `_final_metric` / `history.npz` mapping unchanged)
