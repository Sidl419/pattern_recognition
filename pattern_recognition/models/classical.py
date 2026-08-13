from __future__ import annotations

from typing import Any, Self

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, NuSVC


def positive_class_scores(estimator: Any, x: np.ndarray) -> np.ndarray:
    """P(y=1) when calibrated probabilities exist, else the SVM margin."""
    x = np.asarray(x, dtype=np.float64)
    if _has_working_predict_proba(estimator):
        return estimator.predict_proba(x)[:, 1].astype(np.float32)
    return estimator.decision_function(x).astype(np.float32)


def _has_working_predict_proba(estimator: Any) -> bool:
    if isinstance(estimator, (SVC, NuSVC)):
        return getattr(estimator, "probability", False) is True
    return hasattr(estimator, "predict_proba")


class SklearnSVM:
    """Binary flash scorer: ``SVC`` plus optional Platt calibration.

    ``probability=True`` (the default) wraps ``SVC`` in
    ``CalibratedClassifierCV(..., method="sigmoid", ensemble=False)`` — sklearn's
    replacement for the deprecated ``SVC(probability=True)`` libsvm Platt path.
    ``probability=False`` fits a bare ``SVC`` and scores with ``decision_function``.
    """

    def __init__(self, **svc_kwargs: Any) -> None:
        params = {
            "C": 1.0,
            "kernel": "rbf",
            "probability": True,
            **svc_kwargs,
        }
        self.svc_kwargs = params
        self._clf: BaseEstimator | None = None

    @property
    def fitted_estimator(self) -> BaseEstimator:
        if self._clf is None:
            raise RuntimeError("SklearnSVM is not fitted")
        return self._clf

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        params = dict(self.svc_kwargs)
        want_proba = bool(params.pop("probability", True))
        svc = SVC(**params)
        clf: BaseEstimator
        if want_proba:
            clf = CalibratedClassifierCV(svc, method="sigmoid", ensemble=False)
        else:
            clf = svc
        clf.fit(X, y)
        self._clf = clf
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return positive_class_scores(self.fitted_estimator, X)
