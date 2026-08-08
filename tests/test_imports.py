def test_package_imports():
    from pattern_recognition.training.metrics import compute_itr
    from pattern_recognition.models.cnn import BaseCNN, EEGNet
    from pattern_recognition.data.time_shift import build_timeshifted_dataset
    from pattern_recognition.losses import GraphLoss
    # Wolpaw ITR at chance (P=0.5, N=2) is exactly 0; use above-chance accuracy.
    assert compute_itr(0.75, 2) > 0
