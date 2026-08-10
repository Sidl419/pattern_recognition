from pattern_recognition.training.metrics import compute_itr


def test_compute_itr_binary_edges():
    assert compute_itr(1.0, 2) > 0
    assert abs(compute_itr(0.5, 2)) < 1e-6
