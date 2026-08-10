import json

from pattern_recognition.experiment import run_experiment


def test_contextual_transformer_smoke_run(tmp_path):
    cfg = {
        "name": "ct_smoke",
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
    }
    run_dir = run_experiment(cfg)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["model_mode"] == "flash_scorer"
    assert (run_dir / "model.pt").is_file()


def test_sequence_classifier_smoke_run(tmp_path):
    cfg = {
        "name": "sc_smoke",
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
                "include_character_head": True,
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
    run_dir = run_experiment(cfg)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["model_mode"] == "selection_classifier"
