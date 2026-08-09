import pytest
from pydantic import ValidationError

from pattern_recognition.speller.schema import SpellerBenchmarkConfig

SAMARA_OK = {
    "tag": "samara_sim_within_r10",
    "model_mode": "flash_scorer",
    "protocol": "samara_single_flash_sim",
    "subject_mode": "within_subject",
    "test_subjects": None,
    "repetitions": [1, 2, 5, 10],
    "online": {"early_stop": False, "margin_tau": None},
    "run_dir": "results/binary_run/",
    "split": {
        "seed": 0,
        "epoch_holdout": 0.3,
        "stratify": True,
        "val_fraction": 0.2,
    },
    "simulation": {
        "seed": 0,
        "phrase": "JUST_DO_IT",
    },
    "allow_split_mismatch": False,
    "allow_train_pool_eval": False,
}

BCI3_WITHIN_OK = {
    "tag": "bci3_within_r15",
    "model_mode": "flash_scorer",
    "protocol": "bci3_rowcol",
    "subject_mode": "within_subject",
    "test_subjects": None,
    "repetitions": [1, 2, 5, 10, 15],
    "online": {"early_stop": False, "margin_tau": None},
    "run_dir": "results/binary_run/",
    "split": None,
    "simulation": None,
}

BCI3_CROSS_OK = {
    **BCI3_WITHIN_OK,
    "tag": "bci3_A_to_B_r15",
    "subject_mode": "cross_subject",
    "test_subjects": ["B"],
}


def test_samara_ok():
    cfg = SpellerBenchmarkConfig.model_validate(SAMARA_OK)
    assert cfg.protocol == "samara_single_flash_sim"
    assert cfg.simulation is not None
    assert cfg.simulation.phrase == "JUST_DO_IT"


def test_bci3_null_split_sim_ok():
    cfg = SpellerBenchmarkConfig.model_validate(BCI3_WITHIN_OK)
    assert cfg.split is None
    assert cfg.simulation is None


def test_bci3_cross_with_test_subjects_ok():
    cfg = SpellerBenchmarkConfig.model_validate(BCI3_CROSS_OK)
    assert cfg.subject_mode == "cross_subject"
    assert cfg.test_subjects == ["B"]


def test_bci3_rejects_epoch_holdout():
    bad = {
        **BCI3_WITHIN_OK,
        "split": {
            "seed": 0,
            "epoch_holdout": 0.3,
            "stratify": False,
            "val_fraction": 0.2,
        },
    }
    with pytest.raises(ValidationError, match="epoch_holdout"):
        SpellerBenchmarkConfig.model_validate(bad)


def test_bci3_rejects_stratify_when_split_set():
    bad = {
        **BCI3_WITHIN_OK,
        "split": {
            "seed": 0,
            "epoch_holdout": 0.0,
            "stratify": True,
            "val_fraction": 0.2,
        },
    }
    with pytest.raises(ValidationError, match="stratify"):
        SpellerBenchmarkConfig.model_validate(bad)


def test_bci3_rejects_simulation_phrase():
    bad = {
        **BCI3_WITHIN_OK,
        "simulation": {"phrase": "HELLO"},
    }
    with pytest.raises(ValidationError, match="simulation"):
        SpellerBenchmarkConfig.model_validate(bad)


def test_samara_requires_split():
    bad = {**SAMARA_OK, "split": None}
    with pytest.raises(ValidationError, match="split"):
        SpellerBenchmarkConfig.model_validate(bad)


def test_samara_requires_phrase():
    bad = {**SAMARA_OK, "simulation": {"seed": 0, "phrase": ""}}
    with pytest.raises(ValidationError, match="phrase"):
        SpellerBenchmarkConfig.model_validate(bad)


def test_samara_rejects_phrase_not_in_grid():
    bad = {**SAMARA_OK, "simulation": {"phrase": "XYZ"}}
    with pytest.raises(ValidationError, match="grid"):
        SpellerBenchmarkConfig.model_validate(bad)


def test_subject_mode_inside_split_rejected():
    bad = {
        **SAMARA_OK,
        "split": {
            "seed": 0,
            "epoch_holdout": 0.3,
            "stratify": True,
            "val_fraction": 0.2,
            "subject_mode": "within_subject",
        },
    }
    with pytest.raises(ValidationError):
        SpellerBenchmarkConfig.model_validate(bad)


def test_test_subjects_inside_split_rejected():
    bad = {
        **SAMARA_OK,
        "split": {
            "seed": 0,
            "epoch_holdout": 0.3,
            "stratify": True,
            "val_fraction": 0.2,
            "test_subjects": ["B"],
        },
    }
    with pytest.raises(ValidationError):
        SpellerBenchmarkConfig.model_validate(bad)


def test_within_subject_rejects_test_subjects():
    bad = {**BCI3_WITHIN_OK, "test_subjects": ["B"]}
    with pytest.raises(ValidationError, match="test_subjects"):
        SpellerBenchmarkConfig.model_validate(bad)


def test_cross_subject_requires_test_subjects():
    bad = {**BCI3_CROSS_OK, "test_subjects": None}
    with pytest.raises(ValidationError, match="test_subjects"):
        SpellerBenchmarkConfig.model_validate(bad)


def test_early_stop_requires_tau():
    bad = {
        **BCI3_WITHIN_OK,
        "online": {"early_stop": True, "margin_tau": None},
    }
    with pytest.raises(ValidationError, match="margin_tau"):
        SpellerBenchmarkConfig.model_validate(bad)


def test_early_stop_rejects_non_positive_tau():
    bad = {
        **BCI3_WITHIN_OK,
        "online": {"early_stop": True, "margin_tau": 0.0},
    }
    with pytest.raises(ValidationError, match="margin_tau"):
        SpellerBenchmarkConfig.model_validate(bad)
