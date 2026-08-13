import numpy as np
import pytest
from pattern_recognition.experiment import run_experiment
from pattern_recognition.selections.types import Selection
from pattern_recognition.speller.benchmark import (
    ContextualFlashScorer,
    load_flash_scorer_from_run,
)

from tests.speller.helpers import synthetic_sequence_experiment_config


@pytest.fixture(scope="module")
def ct_run_dir(tmp_path_factory):
    """One CT train for this module — load/prefix/mismatch share the checkpoint."""
    return run_experiment(
        synthetic_sequence_experiment_config(
            tmp_path_factory.mktemp("ct"),
            model_name="ContextualTransformer",
            name="ct_scorer",
        )
    )


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


def test_contextual_flash_scorer_rejects_protocol_mismatch(ct_run_dir):
    with pytest.raises(ValueError, match="protocol"):
        load_flash_scorer_from_run(
            ct_run_dir,
            model_mode="flash_scorer",
            protocol="samara_single_flash_sim",
        )


def test_load_contextual_flash_scorer_from_run(ct_run_dir):
    scorer = load_flash_scorer_from_run(ct_run_dir, model_mode="flash_scorer")
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


def test_contextual_prefix_r_scores_differ_from_full_context(ct_run_dir):
    """Prefix-r packing must not leak future-repeat flashes into early logits."""
    scorer = load_flash_scorer_from_run(ct_run_dir, model_mode="flash_scorer")
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
    assert not np.allclose(at_r1[prefix], full[prefix])


def test_evaluate_selections_calls_score_fn_per_r():
    """score_fn path re-scores at each r (no CT train — contract is the loop)."""
    from pattern_recognition.selections.grids import BCI3_GRID
    from pattern_recognition.selections.protocols import get_protocol
    from pattern_recognition.speller.metrics import evaluate_selections

    selection = _two_repeat_selection(np.random.default_rng(1))
    protocol = get_protocol("bci3_rowcol")
    calls: list[int] = []

    def score_fn(sel, r):
        calls.append(r)
        n = len(sel.flashes)
        scores = np.zeros(n, dtype=np.float64)
        scores[sel.repeat_index < r] = 1.0
        return scores

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
