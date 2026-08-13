import numpy as np
from pattern_recognition.selections.types import Selection
from pattern_recognition.speller.benchmark import SklearnFlashScorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC


def test_sklearn_flash_scorer_uses_calibrated_predict_proba():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 64))
    y = rng.integers(0, 2, size=20)
    clf = CalibratedClassifierCV(SVC(kernel="linear"), ensemble=False).fit(X, y)
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
    expected = clf.predict_proba(flashes.reshape(5, -1))[:, 1].astype(np.float32)
    np.testing.assert_allclose(scores, expected, rtol=0, atol=0)
