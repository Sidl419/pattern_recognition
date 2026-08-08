from pattern_recognition.training.metrics import compute_itr


def test_itr_perfect_binary():
    assert compute_itr(1.0, 2) > 0


def test_itr_chance_near_zero():
    assert abs(compute_itr(0.5, 2)) < 1e-6
