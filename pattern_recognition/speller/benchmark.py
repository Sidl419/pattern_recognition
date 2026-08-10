"""Speller benchmark runner — writes artifacts under ``run_dir/speller/<tag>/``."""

from __future__ import annotations

import csv
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

import pattern_recognition.models  # noqa: F401 — register models
import pattern_recognition.speller.protocols  # noqa: F401 — register protocols
from pattern_recognition.experiment.schema import ExperimentConfig
from pattern_recognition.models import get_model
from pattern_recognition.training.device import resolve_device
from pattern_recognition.reporting.plots import save_speller_plots
from pattern_recognition.speller.grids import COL_CODE, ROW_CODE
from pattern_recognition.speller.metrics import evaluate_selections
from pattern_recognition.speller.online import DecodeMode
from pattern_recognition.speller.protocols import get_protocol
from pattern_recognition.speller.protocols.base import SpellerProtocol
from pattern_recognition.speller.schema import SpellerBenchmarkConfig
from pattern_recognition.speller.types import Selection


@runtime_checkable
class FlashScorer(Protocol):
    """Score individual flashes within a selection."""

    def predict_scores(self, selection: Selection) -> np.ndarray:
        """Return one score per flash in ``selection``."""
        ...


class OracleFlashScorer:
    """Perfect oracle: 1 on flashes for the target symbol, 0 elsewhere."""

    def __init__(self, grid, *, mode: DecodeMode = "single_flash") -> None:
        self._grid = grid
        self._mode = mode

    def predict_scores(self, selection: Selection) -> np.ndarray:
        if self._mode == "rowcol":
            target_row = ROW_CODE[selection.target_char]
            target_col = COL_CODE[selection.target_char]
            return np.where(
                (selection.stimulus_ids == target_row)
                | (selection.stimulus_ids == target_col),
                1.0,
                0.0,
            )
        target_cell = self._grid.index_of(selection.target_char)
        return np.where(selection.stimulus_ids == target_cell, 1.0, 0.0)


class RunFlashScorer:
    """Score flashes using a binary CNN checkpoint from ``run_dir``."""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self._model = model
        self._device = device

    def predict_scores(self, selection: Selection) -> np.ndarray:
        flashes = np.asarray(selection.flashes, dtype=np.float32)
        if flashes.ndim == 1:
            flashes = flashes[np.newaxis, ...]
        x = torch.from_numpy(flashes).to(self._device)
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(x)
        if outputs.ndim == 2 and outputs.shape[-1] >= 2:
            scores = outputs[:, 1]
        else:
            scores = outputs.reshape(outputs.shape[0], -1)[:, 0]
        return scores.detach().cpu().numpy()


def load_flash_scorer_from_run(
    run_dir: Path,
    *,
    model_mode: str,
) -> FlashScorer:
    """Load a flash scorer checkpoint from a binary experiment run."""
    if model_mode == "selection_classifier":
        raise NotImplementedError(
            "selection_classifier scoring from run_dir is not implemented in v1; "
            "inject a scores_provider or train a selection-classifier checkpoint"
        )
    if model_mode != "flash_scorer":
        raise ValueError(f"Unsupported model_mode {model_mode!r}")

    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Binary run config not found: {config_path}"
        )

    checkpoint_path = run_dir / "model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No model checkpoint at {checkpoint_path}; "
            "train with train.save_model=true or inject scores_provider"
        )

    exp_cfg = ExperimentConfig.model_validate(
        json.loads(config_path.read_text())
    )
    device_requested = exp_cfg.device
    meta_path = run_dir / "run_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        device_requested = meta.get("device_requested", device_requested)

    _, device = resolve_device(device_requested)
    model = get_model(exp_cfg.model.name)(**exp_cfg.model.params)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return RunFlashScorer(model, device)


def _load_config(
    config: SpellerBenchmarkConfig | dict | str | Path,
) -> SpellerBenchmarkConfig:
    if isinstance(config, SpellerBenchmarkConfig):
        return config
    if isinstance(config, dict):
        return SpellerBenchmarkConfig.model_validate(config)
    path = Path(config)
    payload = json.loads(path.read_text())
    return SpellerBenchmarkConfig.model_validate(payload)


def _decode_mode(protocol: SpellerProtocol) -> DecodeMode:
    if protocol.name == "bci3_rowcol":
        return "rowcol"
    return "single_flash"


def _resolve_speller_dir(run_dir: Path, tag: str) -> Path:
    base = run_dir / "speller" / tag
    if not base.exists():
        return base
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return run_dir / "speller" / f"{tag}_{stamp}"


def _simulation_seed(cfg: SpellerBenchmarkConfig) -> int:
    if cfg.simulation is not None and cfg.simulation.seed is not None:
        return cfg.simulation.seed
    if cfg.split is not None:
        return cfg.split.seed
    return 0


def _split_seed(cfg: SpellerBenchmarkConfig) -> int | None:
    return cfg.split.seed if cfg.split is not None else None


def _validate_binary_split(cfg: SpellerBenchmarkConfig, run_dir: Path) -> None:
    if cfg.allow_split_mismatch:
        return
    binary_cfg_path = run_dir / "config.json"
    if not binary_cfg_path.is_file():
        return
    binary_cfg = json.loads(binary_cfg_path.read_text())
    binary_split = binary_cfg.get("split")
    if binary_split is None or cfg.split is None:
        return
    speller_split = cfg.split.model_dump()
    if binary_split != speller_split:
        raise ValueError(
            "Speller split block does not match binary run config.json split; "
            "set allow_split_mismatch=true to override"
        )


def _load_binary_config(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _build_synthetic_selections(
    cfg: SpellerBenchmarkConfig, protocol: SpellerProtocol
) -> list[Selection]:
    r_max = max(cfg.repetitions)
    if cfg.protocol == "samara_single_flash_sim":
        assert cfg.simulation is not None
        n_cells = protocol.grid.n_rows * protocol.grid.n_cols
        phrase_len = len(cfg.simulation.phrase)
        holdout = cfg.split.epoch_holdout if cfg.split is not None else 0.3
        min_eval_epochs = phrase_len * n_cells * r_max
        n_epochs = max(200, int(min_eval_epochs / max(holdout, 1e-6) * 2))
        kwargs: dict = {
            "phrase": cfg.simulation.phrase,
            "r_max": r_max,
            "seed": _simulation_seed(cfg),
            "n_epochs": n_epochs,
        }
        if cfg.split is not None:
            kwargs["epoch_holdout"] = cfg.split.epoch_holdout
        return protocol.build_synthetic_selections(**kwargs)

    subject = "A"
    if cfg.test_subjects:
        subject = cfg.test_subjects[0]
    return protocol.build_synthetic_selections(
        phrase="AB",
        r_max=r_max,
        subject=subject,
        seed=_simulation_seed(cfg),
    )


def _build_real_selections(
    cfg: SpellerBenchmarkConfig,
    protocol: SpellerProtocol,
    run_dir: Path,
) -> list[Selection]:
    from pattern_recognition.speller.data_loading import (
        build_bci3_selections_from_mat,
        build_samara_selections_from_dir,
        merge_protocol_params,
    )

    params = merge_protocol_params(
        _load_binary_config(run_dir), cfg.protocol_params
    )
    r_max = max(cfg.repetitions)

    if cfg.protocol == "samara_single_flash_sim":
        assert cfg.simulation is not None
        assert cfg.split is not None or cfg.allow_train_pool_eval
        data_path = params.get("path")
        if not data_path:
            raise FileNotFoundError(
                "Samara real-data benchmark requires data path: set "
                "binary run config data.params.path or "
                "speller protocol_params.path (e.g. Samara_data/). "
                "For CI-only random flashes set use_synthetic=true."
            )
        holdout = (
            cfg.split.epoch_holdout
            if cfg.split is not None
            else float(params.get("val_fraction", 0.2))
        )
        seed = cfg.split.seed if cfg.split is not None else int(params.get("seed", 0))
        stratify = cfg.split.stratify if cfg.split is not None else True
        subjects = None
        if cfg.subject_mode == "within_subject" and params.get("subject"):
            subjects = [str(params["subject"])]
        elif cfg.test_subjects:
            subjects = list(cfg.test_subjects)
        return build_samara_selections_from_dir(
            data_path,
            phrase=cfg.simulation.phrase,
            r_max=r_max,
            epoch_holdout=holdout,
            seed=seed,
            stratify=stratify,
            channel_idx=int(params.get("channel_idx", 1)),
            epoch_len=int(params.get("epoch_len", 250)),
            file_pattern=str(
                params.get("file_pattern", "S*-P300_classic.mat")
            ),
            subjects=subjects,
            simulation_seed=_simulation_seed(cfg),
        )

    # bci3_rowcol
    test_mat = params.get("test_mat") or params.get("mat_path")
    eloc_path = params.get("eloc_path")
    if not test_mat or not eloc_path:
        raise FileNotFoundError(
            "BCI3 real-data benchmark requires test_mat and eloc_path via "
            "binary run config data.params or speller protocol_params. "
            "For CI-only random flashes set use_synthetic=true."
        )
    subject = str(params.get("subject", "A")).upper()
    if cfg.test_subjects:
        subject = str(cfg.test_subjects[0]).upper()
    return build_bci3_selections_from_mat(
        test_mat,
        eloc_path=eloc_path,
        subject=subject,
        channel_name=str(params.get("channel_name", "Pz")),
        n_channels=int(params.get("n_channels", 64)),
        sfreq=float(params.get("sfreq", 120.0)),
        sample_size=int(params.get("sample_size", 72)),
        apply_filter=bool(params.get("filter", True)),
        test_chars=params.get("test_chars"),
        max_repetitions=r_max,
        max_chars=params.get("max_chars"),
    )


def _build_selections(
    cfg: SpellerBenchmarkConfig,
    protocol: SpellerProtocol,
    run_dir: Path,
) -> list[Selection]:
    if cfg.use_synthetic:
        return _build_synthetic_selections(cfg, protocol)
    return _build_real_selections(cfg, protocol, run_dir)


def _resolve_scorer(
    cfg: SpellerBenchmarkConfig,
    run_dir: Path,
    scores_provider: FlashScorer | None,
) -> FlashScorer:
    if scores_provider is not None:
        return scores_provider
    if cfg.model_mode == "selection_classifier":
        raise NotImplementedError(
            "selection_classifier scoring without an injected provider is not "
            "implemented in v1"
        )
    return load_flash_scorer_from_run(run_dir, model_mode=cfg.model_mode)


def _per_subject_rows(
    predictions: list[dict], repetitions: list[int]
) -> list[dict]:
    subjects = sorted(
        {row["subject"] for row in predictions if row.get("subject") is not None}
    )
    if not subjects:
        subjects = [None]

    rows: list[dict] = []
    for subject in subjects:
        for r in repetitions:
            subset = [
                row
                for row in predictions
                if row["r"] == r
                and (subject is None or row.get("subject") == subject)
            ]
            if not subset:
                char_acc = 0.0
            else:
                char_acc = float(
                    np.mean([row["pred"] == row["true"] for row in subset])
                )
            row_out = {"subject": subject or "", "r": r, "char_acc": char_acc}
            rows.append(row_out)
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_speller_benchmark(
    config: SpellerBenchmarkConfig | dict | str | Path,
    *,
    scores_provider: FlashScorer | None = None,
) -> Path:
    """Run speller evaluation and write artifacts under ``run_dir/speller/<tag>/``."""
    cfg = _load_config(config)
    run_dir = Path(cfg.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    if cfg.allow_train_pool_eval:
        warnings.warn(
            "allow_train_pool_eval=true: evaluation may use train-pool epochs",
            stacklevel=2,
        )

    _validate_binary_split(cfg, run_dir)

    protocol = get_protocol(cfg.protocol)
    mode = _decode_mode(protocol)
    scorer = _resolve_scorer(cfg, run_dir, scores_provider)

    started_at = datetime.now(timezone.utc)
    selections = _build_selections(cfg, protocol, run_dir)
    scores_per_sel = [scorer.predict_scores(sel) for sel in selections]

    eval_result = evaluate_selections(
        selections,
        scores_per_sel,
        protocol.decode,
        cfg.repetitions,
        n_classes=protocol.grid.n_rows * protocol.grid.n_cols,
        soa_s=protocol.soa_s,
        flashes_per_repeat=protocol.flashes_per_repeat,
        mode=mode,
        grid=protocol.grid,
        early_stop=cfg.online.early_stop,
        margin_tau=cfg.online.margin_tau,
    )

    speller_dir = _resolve_speller_dir(run_dir, cfg.tag)
    speller_dir.mkdir(parents=True, exist_ok=False)

    finished_at = datetime.now(timezone.utc)

    (speller_dir / "config.json").write_text(
        cfg.model_dump_json(indent=2) + "\n"
    )

    meta = {
        "tag": cfg.tag,
        "model_mode": cfg.model_mode,
        "protocol": cfg.protocol,
        "label_source": protocol.label_source,
        "subject_mode": cfg.subject_mode,
        "use_synthetic": cfg.use_synthetic,
        "split_seed": _split_seed(cfg),
        "simulation_seed": _simulation_seed(cfg),
        "allow_split_mismatch": cfg.allow_split_mismatch,
        "allow_train_pool_eval": cfg.allow_train_pool_eval,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "n_selections": len(selections),
    }
    (speller_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    speller_metrics = {
        "acc_vs_repeats": eval_result["acc_vs_repeats"],
        "n_selections": len(selections),
        "repetitions": cfg.repetitions,
    }
    if "early_stop" in eval_result:
        speller_metrics["early_stop"] = eval_result["early_stop"]
    (speller_dir / "speller_metrics.json").write_text(
        json.dumps(speller_metrics, indent=2) + "\n"
    )

    _write_csv(
        speller_dir / "acc_vs_repeats.csv",
        ["r", "char_acc", "itr"],
        eval_result["acc_vs_repeats"],
    )

    per_subject = _per_subject_rows(eval_result["predictions"], cfg.repetitions)
    for row in per_subject:
        r = row["r"]
        acc_point = next(
            point for point in eval_result["acc_vs_repeats"] if point["r"] == r
        )
        row["itr"] = acc_point["itr"]
    _write_csv(
        speller_dir / "per_subject.csv",
        ["subject", "r", "char_acc", "itr"],
        per_subject,
    )

    prediction_rows = [
        {
            "subject": row.get("subject", ""),
            "selection_id": row["selection_id"],
            "r": row["r"],
            "true": row["true"],
            "pred": row["pred"],
        }
        for row in eval_result["predictions"]
    ]
    _write_csv(
        speller_dir / "predictions.csv",
        ["subject", "selection_id", "r", "true", "pred"],
        prediction_rows,
    )

    if cfg.plots:
        save_speller_plots(speller_dir)

    return speller_dir
