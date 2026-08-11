import pytest
from pattern_recognition.experiment import run_experiment
from pattern_recognition.speller.benchmark import (
    load_flash_scorer_from_run,
    load_selection_classifier_from_run,
    run_speller_benchmark,
)

from tests.speller.helpers import synthetic_sequence_experiment_config


@pytest.fixture(scope="module")
def sc_run_dir(tmp_path_factory):
    """One SC train for gate tests in this module."""
    return run_experiment(
        synthetic_sequence_experiment_config(
            tmp_path_factory.mktemp("sc"),
            model_name="SequenceClassifier",
            name="sc_gates",
        )
    )


@pytest.fixture(scope="module")
def binary_run_dir(tmp_path_factory):
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
            "output_dir": str(tmp_path_factory.mktemp("bin")),
        }
    )


def test_selection_classifier_benchmark_smoke(tmp_path):
    run_dir = run_experiment(
        synthetic_sequence_experiment_config(
            tmp_path,
            model_name="SequenceClassifier",
            name="sc_bench",
        )
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


def test_sc_checkpoint_rejects_flash_scorer_mode(sc_run_dir):
    with pytest.raises(ValueError, match="selection_classifier"):
        load_flash_scorer_from_run(sc_run_dir, model_mode="flash_scorer")


def test_binary_checkpoint_rejects_selection_classifier_mode(binary_run_dir):
    with pytest.raises(ValueError, match="SequenceClassifier"):
        load_selection_classifier_from_run(binary_run_dir)


def test_selection_classifier_rejects_early_stop(sc_run_dir):
    with pytest.raises(ValueError, match="early_stop"):
        run_speller_benchmark(
            {
                "tag": "sc_early",
                "model_mode": "selection_classifier",
                "protocol": "bci3_rowcol",
                "repetitions": [1],
                "run_dir": str(sc_run_dir),
                "use_synthetic": True,
                "plots": False,
                "online": {"early_stop": True, "margin_tau": 1.0},
                "protocol_params": {
                    "phrase": "A",
                    "r_max": 1,
                    "flash_shape": [1, 64],
                },
            }
        )


def test_selection_classifier_rejects_protocol_mismatch(sc_run_dir):
    with pytest.raises(ValueError, match="protocol"):
        load_selection_classifier_from_run(
            sc_run_dir, protocol="samara_single_flash_sim"
        )
