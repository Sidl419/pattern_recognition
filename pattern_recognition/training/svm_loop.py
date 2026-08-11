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
    if model.svc_kwargs.get("probability", False):
        y_hat = (scores >= 0.5).astype(int)
    else:
        y_hat = (scores >= 0.0).astype(int)
    acc = float(accuracy_score(y_va, y_hat))
    bacc = float(balanced_accuracy_score(y_va, y_hat))
    f1 = float(f1_score(y_va, y_hat, zero_division=0))
    itr = float(compute_itr(acc, n_classes=2))
    elapsed = float(time.time() - started)
    acc_dict = {
        "Accuracy": np.array([acc], dtype=float),
        "Balanced Accuracy": np.array([bacc], dtype=float),
        "F1-score": np.array([f1], dtype=float),
        "ITR": np.array([itr], dtype=float),
    }
    val_loss_history = [float("nan")]
    return val_loss_history, acc_dict, elapsed
