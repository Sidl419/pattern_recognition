import pytest
from pattern_recognition.experiment.schema import ExperimentConfig

VALID = {
    "name": "smoke",
    "seed": 0,
    "device": "cpu",
    "data": {
        "pipeline": "SyntheticBinary",
        "params": {"n_train": 32, "n_val": 16, "n_channels": 1, "n_times": 64},
    },
    "model": {"name": "BaseCNN", "params": {"input_feat_dim": 64, "n_channels": 1}},
    "train": {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 8,
        "num_epochs": 1,
        "step_size": 1,
        "gamma": 1.0,
        "save_model": False,
    },
    "output_dir": "results/",
}


def test_valid_config_parses():
    cfg = ExperimentConfig.model_validate(VALID)
    assert cfg.name == "smoke"


def test_invalid_device_rejected():
    bad = {**VALID, "device": "tpu"}
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(bad)
