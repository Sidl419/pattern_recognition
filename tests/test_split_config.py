"""Shared binary/speller split config and three-way epoch partitioning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pattern_recognition.data.splits import three_way_epoch_split
from pattern_recognition.experiment.schema import ExperimentConfig, SplitConfig
from pattern_recognition.speller.benchmark import run_speller_benchmark
from pattern_recognition.speller.schema import SplitConfig as SpellerSplitConfig
from pattern_recognition.speller.simulate import stratified_epoch_holdout

VALID_EXPERIMENT = {
    "name": "smoke",
    "seed": 0,
    "device": "cpu",
    "data": {
        "pipeline": "SyntheticBinary",
        "params": {
            "n_train": 32,
            "n_val": 16,
            "n_channels": 1,
            "n_times": 64,
        },
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


def test_experiment_config_accepts_and_dumps_split():
    cfg = ExperimentConfig.model_validate(
        {
            **VALID_EXPERIMENT,
            "split": {
                "seed": 0,
                "epoch_holdout": 0.3,
                "stratify": True,
                "val_fraction": 0.2,
            },
        }
    )
    assert cfg.split is not None
    assert cfg.split.epoch_holdout == 0.3
    dumped = json.loads(cfg.model_dump_json())
    assert dumped["split"] == {
        "seed": 0,
        "epoch_holdout": 0.3,
        "stratify": True,
        "val_fraction": 0.2,
    }


def test_split_config_shared_between_experiment_and_speller():
    assert SplitConfig is SpellerSplitConfig


def test_three_way_epoch_split_disjoint_and_sizes():
    y = np.array([0] * 70 + [1] * 30)
    train_idx, val_idx, eval_idx = three_way_epoch_split(
        y,
        epoch_holdout=0.3,
        val_fraction=0.2,
        seed=0,
        stratify=True,
    )
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(eval_idx)
    assert set(val_idx).isdisjoint(eval_idx)
    assert len(train_idx) + len(val_idx) + len(eval_idx) == len(y)
    # Holdout ~30% of all epochs
    assert abs(len(eval_idx) / len(y) - 0.3) < 0.08
    # Val is ~20% of the non-holdout pool
    train_pool = len(train_idx) + len(val_idx)
    assert abs(len(val_idx) / train_pool - 0.2) < 0.08


def test_three_way_matches_stratified_holdout_eval():
    y = np.array([0] * 70 + [1] * 30)
    _, _, eval_idx = three_way_epoch_split(
        y, epoch_holdout=0.3, val_fraction=0.2, seed=7, stratify=True
    )
    train_idx, expected_eval = stratified_epoch_holdout(y, 0.3, seed=7, stratify=True)
    assert set(eval_idx.tolist()) == set(expected_eval.tolist())
    assert set(train_idx).isdisjoint(expected_eval)
    assert abs(y[expected_eval].mean() - y.mean()) < 0.15


def test_shared_split_train_eval_indices_disjoint():
    """Binary three-way train/val must not overlap speller eval holdout packing."""
    y = np.array([0] * 160 + [1] * 40)
    epochs = np.random.randn(200, 1, 8)
    seed = 0
    train_idx, val_idx, eval_idx = three_way_epoch_split(
        y,
        epoch_holdout=0.3,
        val_fraction=0.2,
        seed=seed,
        stratify=True,
    )
    _, speller_eval = stratified_epoch_holdout(y, 0.3, seed=seed, stratify=True)
    assert set(eval_idx.tolist()) == set(speller_eval.tolist())
    train_pool = set(train_idx.tolist()) | set(val_idx.tolist())
    assert train_pool.isdisjoint(set(speller_eval.tolist()))

    from pattern_recognition.speller.simulate import simulate_samara_selections

    selections = simulate_samara_selections(
        epochs, y, speller_eval, phrase="JU", r_max=2, seed=1
    )
    used = {i for sel in selections for i in sel.meta["epoch_indices"]}
    assert used.isdisjoint(train_pool)
    assert used.issubset(set(speller_eval.tolist()))


@pytest.mark.parametrize(
    ("train_name", "speller_name"),
    [
        ("samara_contextual_transformer.json", "speller_samara_contextual.json"),
        ("samara_sequence_classifier.json", "speller_samara_sequence_classifier.json"),
        ("samara_pz_eegnet_sc_n1.json", "speller_samara_flash_n1.json"),
        ("samara_pz_basecnn_sc_n1.json", "speller_samara_flash_n1.json"),
        ("samara_pz_svm_sc_n1.json", "speller_samara_flash_n1.json"),
    ],
)
def test_samara_train_example_configs_have_top_level_split(
    train_name: str, speller_name: str
):
    """Train configs must expose split at top level (for speller validation)."""
    root = Path(__file__).resolve().parents[1] / "configs"
    train_payload = json.loads((root / train_name).read_text())
    speller_payload = json.loads((root / speller_name).read_text())

    cfg = ExperimentConfig.model_validate(train_payload)
    assert cfg.split is not None
    assert cfg.split.model_dump() == speller_payload["split"]
    for key in ("seed", "epoch_holdout", "val_fraction", "stratify"):
        assert key not in cfg.data.params


@pytest.mark.parametrize(
    ("train_name", "expected_protocol"),
    [
        ("bci3_contextual_transformer.json", "bci3_rowcol"),
        ("bci3_sequence_classifier.json", "bci3_rowcol"),
        ("samara_contextual_transformer.json", "samara_single_flash_sim"),
        ("samara_sequence_classifier.json", "samara_single_flash_sim"),
    ],
)
def test_ct_sc_train_example_configs_include_data_protocol(
    train_name: str, expected_protocol: str
):
    """CT/SC train configs must dump data.params.protocol for speller loaders."""
    root = Path(__file__).resolve().parents[1] / "configs"
    payload = json.loads((root / train_name).read_text())
    cfg = ExperimentConfig.model_validate(payload)
    assert cfg.data.params.get("protocol") == expected_protocol
    dumped = json.loads(cfg.model_dump_json())
    assert dumped["data"]["params"]["protocol"] == expected_protocol


@pytest.mark.parametrize(
    ("train_name", "expected_epochs"),
    [
        ("samara_pz_eegnet_sc_n1.json", 250),
        ("samara_pz_basecnn_sc_n1.json", 250),
        ("samara_pz_svm_sc_n1.json", 1),
    ],
)
def test_samara_pz_n1_configs_average_and_epochs(train_name: str, expected_epochs: int):
    """Binary n1 configs must use single-trial averaging; CNNs train for 250 epochs."""
    root = Path(__file__).resolve().parents[1] / "configs"
    payload = json.loads((root / train_name).read_text())
    cfg = ExperimentConfig.model_validate(payload)
    assert cfg.data.params.get("n_average") == 1
    assert cfg.train.num_epochs == expected_epochs


def test_samara_speller_rejects_binary_run_without_split(tmp_path: Path):
    run_dir = tmp_path / "binary_run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "name": "no_split",
                "seed": 0,
                "data": {
                    "pipeline": "SamaraWithinSubjectAverage",
                    "params": {"path": "Samara_data/", "n_average": 1},
                },
            }
        )
        + "\n"
    )
    cfg = {
        "tag": "leak_guard",
        "model_mode": "flash_scorer",
        "protocol": "samara_single_flash_sim",
        "subject_mode": "within_subject",
        "repetitions": [1],
        "run_dir": str(run_dir),
        "split": {
            "seed": 0,
            "epoch_holdout": 0.3,
            "stratify": True,
            "val_fraction": 0.2,
        },
        "simulation": {"seed": 0, "phrase": "JU"},
        "use_synthetic": True,
    }
    with pytest.raises(ValueError, match="missing split"):
        run_speller_benchmark(cfg, scores_provider=object())  # type: ignore[arg-type]


def test_flash_scorer_rejects_n_average_not_one(tmp_path: Path):
    run_dir = tmp_path / "binary_run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "name": "avg10",
                "seed": 0,
                "split": {
                    "seed": 0,
                    "epoch_holdout": 0.3,
                    "stratify": True,
                    "val_fraction": 0.2,
                },
                "data": {
                    "pipeline": "SamaraWithinSubjectAverage",
                    "params": {"path": "Samara_data/", "n_average": 10},
                },
            }
        )
        + "\n"
    )
    cfg = {
        "tag": "avg_gate",
        "model_mode": "flash_scorer",
        "protocol": "samara_single_flash_sim",
        "subject_mode": "within_subject",
        "repetitions": [1],
        "run_dir": str(run_dir),
        "split": {
            "seed": 0,
            "epoch_holdout": 0.3,
            "stratify": True,
            "val_fraction": 0.2,
        },
        "simulation": {"seed": 0, "phrase": "JU"},
        "use_synthetic": True,
    }
    with pytest.raises(ValueError, match="n_average"):
        run_speller_benchmark(cfg, scores_provider=object())  # type: ignore[arg-type]
