"""SpellerBenchmarkConfig validation."""

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


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {
                **BCI3_WITHIN_OK,
                "split": {
                    "seed": 0,
                    "epoch_holdout": 0.3,
                    "stratify": False,
                    "val_fraction": 0.2,
                },
            },
            "epoch_holdout",
        ),
        (
            {
                **BCI3_WITHIN_OK,
                "split": {
                    "seed": 0,
                    "epoch_holdout": 0.0,
                    "stratify": True,
                    "val_fraction": 0.2,
                },
            },
            "stratify",
        ),
        (
            {**BCI3_WITHIN_OK, "simulation": {"phrase": "HELLO"}},
            "simulation",
        ),
    ],
)
def test_bci3_rejects_invalid_coupling(payload, match):
    with pytest.raises(ValidationError, match=match):
        SpellerBenchmarkConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({**SAMARA_OK, "split": None}, "split"),
        ({**SAMARA_OK, "simulation": {"seed": 0, "phrase": ""}}, "phrase"),
        ({**SAMARA_OK, "simulation": {"phrase": "XYZ"}}, "grid"),
    ],
)
def test_samara_rejects_invalid_coupling(payload, match):
    with pytest.raises(ValidationError, match=match):
        SpellerBenchmarkConfig.model_validate(payload)


@pytest.mark.parametrize(
    "extra_key_and_value",
    [
        ("subject_mode", "within_subject"),
        ("test_subjects", ["B"]),
    ],
)
def test_split_rejects_subject_policy_keys(extra_key_and_value):
    extra_key, value = extra_key_and_value
    split = {
        "seed": 0,
        "epoch_holdout": 0.3,
        "stratify": True,
        "val_fraction": 0.2,
        extra_key: value,
    }
    with pytest.raises(ValidationError):
        SpellerBenchmarkConfig.model_validate({**SAMARA_OK, "split": split})


def test_within_subject_rejects_test_subjects():
    bad = {**BCI3_WITHIN_OK, "test_subjects": ["B"]}
    with pytest.raises(ValidationError, match="test_subjects"):
        SpellerBenchmarkConfig.model_validate(bad)


def test_cross_subject_requires_test_subjects():
    bad = {**BCI3_CROSS_OK, "test_subjects": None}
    with pytest.raises(ValidationError, match="test_subjects"):
        SpellerBenchmarkConfig.model_validate(bad)


@pytest.mark.parametrize("margin_tau", [None, 0.0])
def test_early_stop_requires_positive_tau(margin_tau):
    bad = {
        **BCI3_WITHIN_OK,
        "online": {"early_stop": True, "margin_tau": margin_tau},
    }
    with pytest.raises(ValidationError, match="margin_tau"):
        SpellerBenchmarkConfig.model_validate(bad)
