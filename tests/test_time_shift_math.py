import numpy as np
from pattern_recognition.data.time_shift import build_timeshifted_dataset, time_shift_info


def test_timeshift_shape_and_bounds():
    raw = np.random.randn(20, 500).astype(np.float32)
    y = np.zeros(20, dtype=np.int64)
    X, yy = build_timeshifted_dataset(raw, y, shift_ms=100, n_channels=3, epoch_len=250, fs=250)
    assert X.shape == (20, 3, 250)
    info = time_shift_info(500, 250, 250, 100, 3)
    assert info["fits_in_raw"] is True


def test_timeshift_raises_if_too_long():
    raw = np.random.randn(5, 500).astype(np.float32)
    y = np.zeros(5, dtype=np.int64)
    import pytest

    with pytest.raises(ValueError):
        build_timeshifted_dataset(raw, y, shift_ms=100, n_channels=20, epoch_len=250, fs=250)
