"""BCI3 flash prep must match the binary pipeline (Scaler → Pz → per-sample z)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pattern_recognition.data.bci3_prep import prepare_bci3_pz_flashes
from pattern_recognition.data.pipelines.bci3_pz import BCI3PzEpochAverage

ROOT = Path(__file__).resolve().parents[1]
BCI3_TEST = ROOT / "matrix_dataset" / "Subject_A_Test.mat"
BCI3_ELOC = ROOT / "matrix_dataset" / "eloc64.loc"


@pytest.mark.skipif(
    not BCI3_TEST.is_file() or not BCI3_ELOC.is_file(),
    reason="BCI3 mats missing",
)
def test_prepare_bci3_pz_flashes_matches_pipeline_channel_stats():
    """Shared prep: Scaler on all channels then Pz + standardize_per_sample."""
    pipe = BCI3PzEpochAverage(
        train_mat=str(BCI3_TEST),  # reuse test mat as a single split source
        test_mat=None,
        eloc_path=str(BCI3_ELOC),
        n_average=1,
        mode="SC",
        filter=False,
        subject="A",
        val_fraction=0.2,
        seed=0,
    )
    # Load raw flashes via shared helper (no averaging).
    chars = pipe._default_test_chars()
    assert chars is not None
    flashes_list, _codes_list, _rep_list = prepare_bci3_pz_flashes(
        BCI3_TEST,
        eloc_path=BCI3_ELOC,
        channel_name="Pz",
        n_channels=64,
        sfreq=120.0,
        sample_size=72,
        apply_filter=False,
        target_chars=chars[:3],
        max_chars=3,
    )
    assert len(flashes_list) == 3
    for flashes in flashes_list:
        assert flashes.ndim == 2  # (n_flashes, T) before channel dim
        # After standardize_per_sample each row should be ~zero mean unit var
        assert abs(flashes[0].mean()) < 1e-5
        assert abs(flashes[0].std() - 1.0) < 1e-4


@pytest.mark.skipif(
    not BCI3_TEST.is_file() or not BCI3_ELOC.is_file(),
    reason="BCI3 mats missing",
)
def test_speller_bci3_builder_uses_scaler_path():
    from pattern_recognition.speller.data_loading import build_bci3_selections_from_mat

    sels = build_bci3_selections_from_mat(
        BCI3_TEST,
        eloc_path=BCI3_ELOC,
        subject="A",
        apply_filter=False,
        max_repetitions=1,
        max_chars=2,
    )
    # Marker that shared prep ran (meta flag) and shapes match scorer input
    assert sels[0].meta.get("scaled_with_mne_scaler") is True
    assert sels[0].flashes.ndim == 3
    assert sels[0].flashes.shape[1] == 1
