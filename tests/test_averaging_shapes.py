import torch
from pattern_recognition.data.averaging import (
    build_multichannel_subject_dataset_unique,
    multichannel_to_single_channel,
)


def test_mc_and_sc_shapes():
    data = torch.randn(100, 250)
    labels = torch.tensor([1] * 50 + [0] * 50)
    X, y = build_multichannel_subject_dataset_unique(data, labels, n_channels=5, seed=0)
    assert X.ndim == 3 and X.shape[1] == 5 and X.shape[2] == 250
    X_sc = multichannel_to_single_channel(X)
    assert X_sc.shape[1] == 1
