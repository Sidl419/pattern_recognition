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
