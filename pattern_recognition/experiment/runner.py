"""Config-driven experiment runner and artifact writer."""

from __future__ import annotations

import inspect
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from pattern_recognition.data.pipelines import get_pipeline
from pattern_recognition.experiment.schema import ExperimentConfig
from pattern_recognition.losses import BrierLoss
from pattern_recognition.models import get_model
from pattern_recognition.selections.packing import collate_selection_packets
from pattern_recognition.training.checkpoint import BestCheckpoint
from pattern_recognition.training.device import resolve_device
from pattern_recognition.training.loop import train_model
from pattern_recognition.training.sequence_loop import (
    train_contextual_transformer,
    train_sequence_classifier,
)

SEQUENCE_MODELS = frozenset({"ContextualTransformer", "SequenceClassifier"})
CLASSICAL_MODELS = frozenset({"SVM"})
BINARY_PACKET_FORBIDDEN = frozenset({"EEGNet", "BaseCNN", "SVM"})


def _is_packet_pipeline(name: str) -> bool:
    return name.endswith("SelectionPackets")


def _load_config(config: ExperimentConfig | dict | str | Path) -> ExperimentConfig:
    if isinstance(config, ExperimentConfig):
        return config
    if isinstance(config, dict):
        return ExperimentConfig.model_validate(config)
    path = Path(config)
    payload = json.loads(path.read_text())
    return ExperimentConfig.model_validate(payload)


def _pipeline_params(cfg: ExperimentConfig) -> dict[str, Any]:
    """Merge data.params with shared ``split`` fields the pipeline accepts."""
    params: dict[str, Any] = {**cfg.data.params}
    explicit_seed = cfg.data.params.get("seed")
    params.setdefault("seed", cfg.seed)
    if cfg.split is None:
        return params

    if explicit_seed is not None and explicit_seed != cfg.split.seed:
        raise ValueError(
            f"Conflicting split seeds: data.params.seed={explicit_seed} and "
            f"split.seed={cfg.split.seed}. Set only one — the split block "
            "governs how epochs are partitioned and is what the speller "
            "benchmark checks against."
        )

    pipeline_cls = get_pipeline(cfg.data.pipeline)
    accepted = set(inspect.signature(pipeline_cls.__init__).parameters)
    split_kwargs = {
        "seed": cfg.split.seed,
        "epoch_holdout": cfg.split.epoch_holdout,
        "val_fraction": cfg.split.val_fraction,
        "stratify": cfg.split.stratify,
    }
    for key, value in split_kwargs.items():
        if key in accepted:
            params[key] = value
    return params


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


def _metric_at(history: np.ndarray | list | torch.Tensor, epoch: int) -> float:
    """Value of a per-epoch history at the selected (best) epoch."""
    arr = np.asarray(history)
    if arr.size == 0:
        return float("nan")
    index = epoch if 0 <= epoch < arr.size else arr.size - 1
    return _as_float(arr[index])


def _create_run_dir(output_dir: Path, name: str, started_at: datetime) -> Path:
    """Create a fresh run directory, never reusing an existing one.

    Timestamps are second-resolution, so back-to-back runs of the same config
    would otherwise land in the same directory and overwrite each other.
    """
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"{name}_{timestamp}"
    suffix = 2
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = output_dir / f"{name}_{timestamp}_{suffix}"
            suffix += 1


def _write_run_meta(run_dir: Path, run_meta: dict[str, Any]) -> None:
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n")


def run_experiment(config: ExperimentConfig | dict | str | Path) -> Path:
    """Validate config, train, write run artifacts, and return the run directory."""
    cfg = _load_config(config)
    device_requested = cfg.device
    device_resolved, device = resolve_device(device_requested)

    started_at = datetime.now(timezone.utc)
    _set_seed(cfg.seed)

    is_sequence = cfg.model.name in SEQUENCE_MODELS
    is_classical = cfg.model.name in CLASSICAL_MODELS
    if is_sequence and not _is_packet_pipeline(cfg.data.pipeline):
        raise ValueError(
            f"Sequence model {cfg.model.name!r} requires a selection-packet "
            f"pipeline (name ending in 'SelectionPackets'); got "
            f"{cfg.data.pipeline!r}"
        )
    if cfg.model.name in BINARY_PACKET_FORBIDDEN and _is_packet_pipeline(
        cfg.data.pipeline
    ):
        raise ValueError(
            f"Binary model {cfg.model.name!r} cannot use selection-packet "
            f"pipeline {cfg.data.pipeline!r} (name ending in 'SelectionPackets'); "
            "use a binary epoch pipeline instead"
        )

    pipeline_cls = get_pipeline(cfg.data.pipeline)
    params = _pipeline_params(cfg)
    bundle = pipeline_cls(**params).build()

    # Write the run record before training so a crash or Ctrl-C hours into a
    # long run still leaves the config and the split behind.
    run_dir = _create_run_dir(Path(cfg.output_dir), cfg.name, started_at)
    (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2) + "\n")
    if bundle.metadata.get("split_indices") is not None:
        (run_dir / "split_indices.json").write_text(
            json.dumps(bundle.metadata["split_indices"], indent=2) + "\n"
        )

    run_meta: dict[str, Any] = {
        "seed": cfg.seed,
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "device_requested": device_requested,
        "device_resolved": device_resolved,
        "name": cfg.name,
        "model_mode": None,
        "status": "running",
    }
    _write_run_meta(run_dir, run_meta)

    learning_params = {
        "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "step_size": cfg.train.step_size,
        "gamma": cfg.train.gamma,
        "num_epochs": cfg.train.num_epochs,
        "model_type": "CNN",
    }
    model = get_model(cfg.model.name)(**cfg.model.params)
    checkpoint = BestCheckpoint(cfg.train.checkpoint_metric)

    try:
        if is_classical:
            from pattern_recognition.training.svm_loop import train_svm

            val_loss_history, acc_dict, time_elapsed = train_svm(
                model, bundle.train, bundle.val
            )
            model_mode = "flash_scorer"
        elif is_sequence:
            dataloaders = {
                "train": DataLoader(
                    bundle.train,
                    batch_size=cfg.train.batch_size,
                    shuffle=True,
                    collate_fn=collate_selection_packets,
                ),
                "val": DataLoader(
                    bundle.val,
                    batch_size=cfg.train.batch_size,
                    shuffle=False,
                    collate_fn=collate_selection_packets,
                ),
            }
            if cfg.model.name == "ContextualTransformer":
                val_loss_history, acc_dict, time_elapsed = train_contextual_transformer(
                    model,
                    dataloaders,
                    learning_params,
                    device=device,
                    checkpoint=checkpoint,
                )
                model_mode = "flash_scorer"
            else:
                val_loss_history, acc_dict, time_elapsed = train_sequence_classifier(
                    model,
                    dataloaders,
                    learning_params,
                    device=device,
                    checkpoint=checkpoint,
                )
                model_mode = "selection_classifier"
        else:
            dataloaders = {
                "train": DataLoader(
                    bundle.train, batch_size=cfg.train.batch_size, shuffle=True
                ),
                "val": DataLoader(
                    bundle.val, batch_size=cfg.train.batch_size, shuffle=False
                ),
            }
            criterion = BrierLoss()
            val_loss_history, acc_dict, time_elapsed = train_model(
                model,
                dataloaders,
                criterion,
                learning_params,
                is_binary=True,
                device=device,
                checkpoint=checkpoint,
            )
            model_mode = "flash_scorer"
    except BaseException as exc:  # includes KeyboardInterrupt on long runs
        run_meta["status"] = "failed"
        run_meta["error"] = f"{type(exc).__name__}: {exc}"
        run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_run_meta(run_dir, run_meta)
        raise

    # Classical training has no epochs; everything else reports (and saved)
    # the epoch the checkpoint tracker selected.
    best_epoch = 0 if is_classical else checkpoint.best_epoch
    checkpoint_metric = "last" if is_classical else checkpoint.metric

    run_meta["model_mode"] = model_mode
    run_meta["status"] = "completed"
    run_meta["checkpoint_metric"] = checkpoint_metric
    run_meta["checkpoint_metric_requested"] = cfg.train.checkpoint_metric
    run_meta["best_epoch"] = int(best_epoch)
    run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_run_meta(run_dir, run_meta)

    metrics = {
        "accuracy": _metric_at(acc_dict["Accuracy"], best_epoch),
        "balanced_accuracy": _metric_at(acc_dict["Balanced Accuracy"], best_epoch),
        "f1": _metric_at(acc_dict["F1-score"], best_epoch),
        "itr": _metric_at(acc_dict["ITR"], best_epoch),
        "brier": _metric_at(acc_dict["Brier"], best_epoch),
        "train_time_sec": float(time_elapsed),
        "best_epoch": int(best_epoch),
        "checkpoint_metric": checkpoint_metric,
        "device_requested": device_requested,
        "device_resolved": device_resolved,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    history_arrays: dict[str, np.ndarray] = {
        "val_loss": np.asarray(val_loss_history, dtype=float),
        "accuracy": np.asarray(acc_dict["Accuracy"], dtype=float),
        "balanced_accuracy": np.asarray(acc_dict["Balanced Accuracy"], dtype=float),
        "f1": np.asarray(acc_dict["F1-score"], dtype=float),
        "itr": np.asarray(acc_dict["ITR"], dtype=float),
        "brier": np.asarray(acc_dict["Brier"], dtype=float),
    }
    np.savez(run_dir / "history.npz", **history_arrays)

    if cfg.train.save_model:
        if is_classical:
            import joblib

            joblib.dump(model.fitted_estimator, run_dir / "model.joblib")
        else:
            torch.save(model.state_dict(), run_dir / "model.pt")

    return run_dir
