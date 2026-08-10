"""Samara selection packing / simulation helpers."""

import numpy as np
from pattern_recognition.speller.simulate import (
    simulate_samara_selections,
    stratified_epoch_holdout,
)


def test_simulate_no_reuse_within_selection_and_seed_stable():
    epochs = np.random.randn(200, 1, 8)
    y = np.array([0] * 160 + [1] * 40)
    _, eval_idx = stratified_epoch_holdout(y, 0.5, seed=0)
    a = simulate_samara_selections(epochs, y, eval_idx, "JU", r_max=2, seed=1)
    b = simulate_samara_selections(epochs, y, eval_idx, "JU", r_max=2, seed=1)
    assert a[0].stimulus_ids.tolist() == b[0].stimulus_ids.tolist()
    idx = a[0].meta["epoch_indices"]
    assert len(idx) == len(set(idx))
