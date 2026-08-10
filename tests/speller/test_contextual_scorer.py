import numpy as np
from pattern_recognition.experiment import run_experiment
from pattern_recognition.speller.benchmark import load_flash_scorer_from_run
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


def test_load_contextual_flash_scorer_from_run(tmp_path):
    run_dir = _ct_run(tmp_path)
    scorer = load_flash_scorer_from_run(run_dir, model_mode="flash_scorer")
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
