import numpy as np
import pattern_recognition.models  # noqa: F401
from pattern_recognition.models import get_model
from pattern_recognition.models.classical import SklearnSVM


def test_svm_registered():
    assert (
        "SVM"
        in __import__(
            "pattern_recognition.models", fromlist=["list_models"]
        ).list_models()
    )


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
