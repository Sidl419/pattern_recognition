import torch
import pytest
from pattern_recognition.speller.decode import decode_from_sequence_output
from pattern_recognition.speller.grids import BCI3_GRID, SAMARA_GRID


def test_decode_rowcol_from_logits():
    row = torch.zeros(6)
    row[0] = 1.0
    col = torch.zeros(6)
    col[0] = 1.0
    out = {"row_logits": row.unsqueeze(0), "column_logits": col.unsqueeze(0)}
    assert decode_from_sequence_output(out, "bci3_rowcol", BCI3_GRID) == BCI3_GRID.char_at(
        0, 0
    )


def test_decode_cell_from_logits():
    logits = torch.zeros(16)
    logits[3] = 1.0
    out = {"character_logits": logits.unsqueeze(0)}
    assert (
        decode_from_sequence_output(out, "samara_single_flash_sim", SAMARA_GRID)
        == SAMARA_GRID.chars[3]
    )


def test_selection_classifier_benchmark_smoke(tmp_path):
    from pattern_recognition.experiment import run_experiment
    from pattern_recognition.speller.benchmark import run_speller_benchmark

    run_dir = run_experiment(
        {
            "name": "sc_bench",
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
                "name": "SequenceClassifier",
                "params": {
                    "eegnet": {"input_feat_dim": 64, "in_channels": 1},
                    "d_model": 32,
                    "nhead": 4,
                    "num_layers": 1,
                    "num_stimulus_codes": 13,
                    "max_flashes": 24,
                    "max_repetitions": 15,
                    "head_mode": "rowcol",
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
        }
    )
    speller_dir = run_speller_benchmark(
        {
            "tag": "sc_syn",
            "model_mode": "selection_classifier",
            "protocol": "bci3_rowcol",
            "repetitions": [1, 2],
            "run_dir": str(run_dir),
            "use_synthetic": True,
            "plots": False,
            "protocol_params": {"phrase": "AB", "r_max": 2, "flash_shape": [1, 64]},
        }
    )
    assert (speller_dir / "speller_metrics.json").is_file()


def _sc_run(tmp_path):
    from pattern_recognition.experiment import run_experiment

    return run_experiment(
        {
            "name": "sc_mismatch",
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
                "name": "SequenceClassifier",
                "params": {
                    "eegnet": {"input_feat_dim": 64, "in_channels": 1},
                    "d_model": 32,
                    "nhead": 4,
                    "num_layers": 1,
                    "num_stimulus_codes": 13,
                    "max_flashes": 24,
                    "max_repetitions": 15,
                    "head_mode": "rowcol",
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
            "output_dir": str(tmp_path / "sc"),
        }
    )


def _binary_run(tmp_path):
    from pattern_recognition.experiment import run_experiment

    return run_experiment(
        {
            "name": "bin_mismatch",
            "seed": 0,
            "device": "cpu",
            "data": {
                "pipeline": "SyntheticBinary",
                "params": {
                    "n_train": 16,
                    "n_val": 8,
                    "n_channels": 1,
                    "n_times": 64,
                },
            },
            "model": {
                "name": "BaseCNN",
                "params": {"input_feat_dim": 64, "n_channels": 1},
            },
            "train": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "batch_size": 8,
                "num_epochs": 1,
                "step_size": 1,
                "gamma": 1.0,
                "save_model": True,
            },
            "output_dir": str(tmp_path / "bin"),
        }
    )


def test_sc_checkpoint_rejects_flash_scorer_mode(tmp_path):
    from pattern_recognition.speller.benchmark import load_flash_scorer_from_run

    run_dir = _sc_run(tmp_path)
    with pytest.raises(ValueError, match="selection_classifier"):
        load_flash_scorer_from_run(run_dir, model_mode="flash_scorer")


def test_binary_checkpoint_rejects_selection_classifier_mode(tmp_path):
    from pattern_recognition.speller.benchmark import (
        load_selection_classifier_from_run,
    )

    run_dir = _binary_run(tmp_path)
    with pytest.raises(ValueError, match="SequenceClassifier"):
        load_selection_classifier_from_run(run_dir)
