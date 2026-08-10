import numpy as np
import pytest
from pattern_recognition.experiment import run_experiment
from pattern_recognition.speller.benchmark import (
    ContextualFlashScorer,
    load_flash_scorer_from_run,
)
from pattern_recognition.speller.types import Selection


def _ct_run(tmp_path):
    return run_experiment({
        "name": "ct_scorer",
        "seed": 0,
        "device": "cpu",
        "data": {
            "pipeline": "SyntheticSelectionPackets",
            "params": {
                "protocol": "bci3_rowcol",
                "n_train": 4,
                "n_val": 2,
                "n_channels": 1,
                "n_times": 64,
                "r_max": 2,
            },
        },
        "model": {
            "name": "ContextualTransformer",
            "params": {
                "eegnet": {"input_feat_dim": 64, "in_channels": 1},
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "num_stimulus_codes": 13,
                "max_flashes": 24,
                "max_repetitions": 15,
            },
        },
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 2,
            "num_epochs": 1,
            "step_size": 1,
            "gamma": 1.0,
            "save_model": True,
        },
        "output_dir": str(tmp_path),
    })


def _two_repeat_selection(rng: np.random.Generator) -> Selection:
    """12 flashes in repeat 0 + 12 in repeat 1 (BCI3 row/col codes 1..12)."""
    n_per = 12
    flashes = rng.standard_normal((2 * n_per, 1, 64)).astype(np.float32)
    stimulus_ids = np.tile(np.arange(1, n_per + 1, dtype=np.int64), 2)
    repeat_index = np.concatenate(
        [
            np.zeros(n_per, dtype=np.int64),
            np.ones(n_per, dtype=np.int64),
        ]
    )
    return Selection(
        flashes=flashes,
        stimulus_ids=stimulus_ids,
        target_char="A",
        repeat_index=repeat_index,
        meta={},
    )


def test_contextual_flash_scorer_rejects_protocol_mismatch(tmp_path):
    run_dir = _ct_run(tmp_path)
    with pytest.raises(ValueError, match="protocol"):
        load_flash_scorer_from_run(
            run_dir,
            model_mode="flash_scorer",
            protocol="samara_single_flash_sim",
        )


def test_load_contextual_flash_scorer_from_run(tmp_path):
    run_dir = _ct_run(tmp_path)
    scorer = load_flash_scorer_from_run(run_dir, model_mode="flash_scorer")
    assert isinstance(scorer, ContextualFlashScorer)
    n = 12
    selection = Selection(
        flashes=np.random.randn(n, 1, 64).astype(np.float32),
        stimulus_ids=np.arange(1, n + 1, dtype=np.int64),
        target_char="A",
        repeat_index=np.zeros(n, dtype=np.int64),
        meta={},
    )
    scores = scorer.predict_scores(selection)
    assert scores.shape == (n,)


def test_contextual_prefix_r_scores_differ_from_full_context(tmp_path):
    """Prefix-r packing must not leak future-repeat flashes into early logits."""
    run_dir = _ct_run(tmp_path)
    scorer = load_flash_scorer_from_run(run_dir, model_mode="flash_scorer")
    assert isinstance(scorer, ContextualFlashScorer)

    rng = np.random.default_rng(0)
    selection = _two_repeat_selection(rng)
    n = len(selection.stimulus_ids)
    prefix = selection.repeat_index < 1

    full = scorer.predict_scores(selection)
    at_r1 = scorer.predict_scores(selection, r=1)

    assert full.shape == (n,)
    assert at_r1.shape == (n,)
    assert np.all(at_r1[~prefix] == 0.0)
    # Overlapping (repeat 0) flashes: truncated CT forward ≠ full-context logits.
    assert not np.allclose(at_r1[prefix], full[prefix])


def test_evaluate_selections_calls_score_fn_per_r(tmp_path):
    """Contextual evaluate path must re-score at each r (not one-shot full context)."""
    from pattern_recognition.speller.grids import BCI3_GRID
    from pattern_recognition.speller.metrics import evaluate_selections
    from pattern_recognition.speller.protocols import get_protocol

    run_dir = _ct_run(tmp_path)
    scorer = load_flash_scorer_from_run(run_dir, model_mode="flash_scorer")
    assert isinstance(scorer, ContextualFlashScorer)

    selection = _two_repeat_selection(np.random.default_rng(1))
    protocol = get_protocol("bci3_rowcol")
    calls: list[int] = []

    def score_fn(sel, r):
        calls.append(r)
        return scorer.predict_scores(sel, r=r)

    result = evaluate_selections(
        [selection],
        None,
        protocol.decode,
        repetitions=[1, 2],
        n_classes=36,
        soa_s=protocol.soa_s,
        flashes_per_repeat=protocol.flashes_per_repeat,
        mode="rowcol",
        grid=BCI3_GRID,
        score_fn=score_fn,
    )

    assert calls == [1, 2]
    assert len(result["predictions"]) == 2
    assert {row["r"] for row in result["predictions"]} == {1, 2}
