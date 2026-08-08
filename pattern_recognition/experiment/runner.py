"""Config-driven experiment runner and artifact writer."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pattern_recognition.data.pipelines import get_pipeline
from pattern_recognition.experiment.schema import ExperimentConfig
from pattern_recognition.models import get_model
from pattern_recognition.training.device import resolve_device
from pattern_recognition.training.loop import train_model


def _load_config(config: ExperimentConfig | dict | str | Path) -> ExperimentConfig:
    if isinstance(config, ExperimentConfig):
        return config
    if isinstance(config, dict):
        return ExperimentConfig.model_validate(config)
    path = Path(config)
    payload = json.loads(path.read_text())
    return ExperimentConfig.model_validate(payload)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _final_metric(history: np.ndarray | list | torch.Tensor) -> float:
    arr = np.asarray(history)
    if arr.size == 0:
        return float("nan")
    return _as_float(arr[-1])


def run_experiment(config: ExperimentConfig | dict | str | Path) -> Path:
    """Validate config, train, write run artifacts, and return the run directory."""
    cfg = _load_config(config)
    device_requested = cfg.device
    device_resolved, device = resolve_device(device_requested)

    started_at = datetime.now(timezone.utc)
    _set_seed(cfg.seed)

    pipeline_cls = get_pipeline(cfg.data.pipeline)
    bundle = pipeline_cls(**cfg.data.params).build()

    dataloaders = {
        "train": DataLoader(
            bundle.train, batch_size=cfg.train.batch_size, shuffle=True
        ),
        "val": DataLoader(
            bundle.val, batch_size=cfg.train.batch_size, shuffle=False
        ),
    }

    model = get_model(cfg.model.name)(**cfg.model.params)
    learning_params = {
        "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "step_size": cfg.train.step_size,
        "gamma": cfg.train.gamma,
        "num_epochs": cfg.train.num_epochs,
        "model_type": "CNN",
    }
    criterion = nn.MSELoss()
    val_loss_history, acc_dict, time_elapsed = train_model(
        model,
        dataloaders,
        criterion,
        learning_params,
        is_binary=True,
        device=device,
    )

    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / f"{cfg.name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        cfg.model_dump_json(indent=2) + "\n"
    )

    finished_at = datetime.now(timezone.utc)
    run_meta = {
        "seed": cfg.seed,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "device_requested": device_requested,
        "device_resolved": device_resolved,
        "name": cfg.name,
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2) + "\n"
    )

    metrics = {
        "accuracy": _final_metric(acc_dict["Accuracy"]),
        "balanced_accuracy": _final_metric(acc_dict["Balanced Accuracy"]),
        "f1": _final_metric(acc_dict["F1-score"]),
        "itr": _final_metric(acc_dict["ITR"]),
        "train_time_sec": float(time_elapsed),
        "device_requested": device_requested,
        "device_resolved": device_resolved,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    history_arrays: dict[str, np.ndarray] = {
        "val_loss": np.asarray(val_loss_history, dtype=float),
        "accuracy": np.asarray(acc_dict["Accuracy"], dtype=float),
        "balanced_accuracy": np.asarray(acc_dict["Balanced Accuracy"], dtype=float),
        "f1": np.asarray(acc_dict["F1-score"], dtype=float),
        "itr": np.asarray(acc_dict["ITR"], dtype=float),
    }
    np.savez(run_dir / "history.npz", **history_arrays)

    if cfg.train.save_model:
        torch.save(model.state_dict(), run_dir / "model.pt")

    return run_dir
