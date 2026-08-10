def test_package_imports():
    from pattern_recognition.training.metrics import compute_itr

    # Wolpaw ITR at chance (P=0.5, N=2) is exactly 0; use above-chance accuracy.
    assert compute_itr(0.75, 2) > 0
