import warnings

import numpy as np
from pattern_recognition.models.classical import SklearnSVM
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC


def test_probability_true_uses_calibrated_svc_not_libsvm_platt():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 32)).astype(np.float64)
    y = rng.integers(0, 2, size=40)
    model = SklearnSVM(C=1.0, kernel="linear", probability=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model.fit(X, y)
    assert isinstance(model.fitted_estimator, CalibratedClassifierCV)
    scores = model.predict_scores(X)
    assert scores.shape == (40,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    expected = model.fitted_estimator.predict_proba(X)[:, 1].astype(np.float32)
    np.testing.assert_allclose(scores, expected, rtol=0, atol=0)


def test_probability_false_uses_raw_svc_margin():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 32)).astype(np.float64)
    y = rng.integers(0, 2, size=40)
    model = SklearnSVM(C=1.0, kernel="linear", probability=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model.fit(X, y)
    assert isinstance(model.fitted_estimator, SVC)
    assert not isinstance(model.fitted_estimator, CalibratedClassifierCV)
    scores = model.predict_scores(X)
    expected = model.fitted_estimator.decision_function(X).astype(np.float32)
    np.testing.assert_allclose(scores, expected, rtol=0, atol=0)
