"""Load real EEG into ``Selection`` objects (BCI3 ground-truth, Samara sim)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

from pattern_recognition.data.bci3_prep import (
    DEFAULT_TEST_A_CHARS,
    DEFAULT_TEST_B_CHARS,
    prepare_bci3_pz_flashes,
)
from pattern_recognition.data.samara import load_samara_subjects
from pattern_recognition.data.splits import stratified_epoch_holdout, subject_seed
from pattern_recognition.selections.grids import SAMARA_GRID
from pattern_recognition.selections.simulate import simulate_samara_selections
from pattern_recognition.selections.types import Selection


def _repeat_index_from_codes(codes: np.ndarray) -> np.ndarray:
    repeat_index = np.zeros(len(codes), dtype=np.int64)
    counts: dict[int, int] = {}
    for i, code in enumerate(codes.astype(int)):
        repeat_index[i] = counts.get(code, 0)
        counts[code] = counts.get(code, 0) + 1
    return repeat_index


def _resolve_bci3_target_chars(
    raw: dict,
    subject: str,
    explicit: list[str] | str | None,
) -> list[str]:
    if explicit is not None:
        return list(explicit) if isinstance(explicit, str) else list(explicit)
    if "TargetChar" in raw:
        value = raw["TargetChar"]
        if isinstance(value, np.ndarray):
            value = value.flat[0]
        return list(str(value))
    if subject.upper() == "A":
        return list(DEFAULT_TEST_A_CHARS)
    if subject.upper() == "B":
        return list(DEFAULT_TEST_B_CHARS)
    raise ValueError(
        "BCI3 target characters unavailable: provide protocol_params.test_chars "
        f"or TargetChar in the .mat (subject={subject!r})"
    )


def build_bci3_selections_from_mat(
    mat_path: str | Path,
    *,
    eloc_path: str | Path,
    subject: str = "A",
    channel_name: str = "Pz",
    n_channels: int = 64,
    sfreq: float = 120.0,
    sample_size: int = 72,
    apply_filter: bool = True,
    test_chars: list[str] | str | None = None,
    max_repetitions: int | None = None,
    max_chars: int | None = None,
) -> list[Selection]:
    """Build ground-truth row×col selections from a BCI3 ``.mat`` file."""
    mat_path = Path(mat_path).expanduser().resolve()
    if not mat_path.is_file():
        raise FileNotFoundError(f"BCI3 .mat not found: {mat_path}")

    raw = sio.loadmat(str(mat_path))
    target_chars = _resolve_bci3_target_chars(raw, subject, test_chars)
    n_chars = len(raw["Flashing"])
    if max_chars is not None:
        n_chars = min(n_chars, int(max_chars))
    if len(target_chars) < n_chars:
        raise ValueError(
            f"Only {len(target_chars)} target chars for {n_chars} character epochs"
        )

    flashes_list, codes_list, _onsets = prepare_bci3_pz_flashes(
        mat_path,
        eloc_path=eloc_path,
        channel_name=channel_name,
        n_channels=n_channels,
        sfreq=sfreq,
        sample_size=sample_size,
        apply_filter=apply_filter,
        target_chars=list(target_chars[:n_chars]),
        max_chars=n_chars,
    )

    selections: list[Selection] = []
    for epoch_num, (flashes_2d, codes) in enumerate(
        zip(flashes_list, codes_list, strict=True)
    ):
        flashes_arr = flashes_2d[:, np.newaxis, :]
        repeat_index = _repeat_index_from_codes(codes)
        if max_repetitions is not None:
            keep = repeat_index < max_repetitions
            flashes_arr = flashes_arr[keep]
            codes = codes[keep]
            repeat_index = repeat_index[keep]

        selections.append(
            Selection(
                flashes=flashes_arr,
                stimulus_ids=codes,
                target_char=str(target_chars[epoch_num]),
                repeat_index=repeat_index,
                meta={
                    "subject": subject.upper(),
                    "label_source": "ground_truth",
                    "char_index": epoch_num,
                    "mat_path": str(mat_path),
                    "scaled_with_mne_scaler": True,
                },
            )
        )
    return selections


def _eval_idx_from_artifact(
    eval_indices: dict[str, list[int]],
    subject: str,
    n_epochs: int,
) -> np.ndarray:
    """Take a subject's holdout epochs from the binary run's ``split_indices``.

    Recomputing the split here instead would silently diverge from training
    whenever the subject set, ordering, or loader params differ.
    """
    if subject not in eval_indices:
        raise KeyError(
            f"Subject {subject!r} has no entry in the binary run's "
            f"split_indices.json (recorded: {sorted(eval_indices)}). The "
            "speller cannot verify its holdout is disjoint from training."
        )
    idx = np.asarray(eval_indices[subject], dtype=np.int64)
    if idx.size == 0:
        raise ValueError(
            f"Binary run recorded an empty eval split for subject {subject!r}; "
            "re-train with split.epoch_holdout in (0, 1)."
        )
    if idx.min() < 0 or idx.max() >= n_epochs:
        raise ValueError(
            f"split_indices.json for subject {subject!r} references epoch "
            f"{int(idx.max())} but only {n_epochs} epochs were loaded. The "
            "speller must use the same channel_idx / epoch_len / file_pattern "
            "and the same .mat files as the binary run."
        )
    return idx


def build_samara_selections_from_dir(
    data_path: str | Path,
    *,
    phrase: str,
    r_max: int,
    epoch_holdout: float,
    seed: int,
    stratify: bool = True,
    channel_idx: int = 1,
    epoch_len: int = 250,
    file_pattern: str = "S*-P300_classic.mat",
    subjects: list[str] | None = None,
    simulation_seed: int | None = None,
    use_train_pool: bool = False,
    eval_indices: dict[str, list[int]] | None = None,
) -> list[Selection]:
    """Load Samara epochs and simulate single-flash selections on holdout pools.

    ``eval_indices`` are the per-subject holdout epochs recorded by the binary
    run (``split_indices.json``) and are the only leakage-free source: the
    fallback recomputation below reproduces the training split only when the
    subject set and ordering match exactly.

    When ``use_train_pool`` is true (exploratory), every epoch is eligible for
    packing — including epochs that would be in the binary train pool.
    """
    data, labels, subject_ids = load_samara_subjects(
        data_path,
        channel_idx=channel_idx,
        file_pattern=file_pattern,
        subjects=subjects,
    )
    data_path = Path(data_path).expanduser().resolve()

    sim_seed = seed if simulation_seed is None else simulation_seed
    if use_train_pool or epoch_holdout <= 0.0:
        eval_source = "train_pool"
    elif eval_indices is not None:
        eval_source = "split_indices"
    else:
        eval_source = "recomputed"

    selections: list[Selection] = []
    for subj in subject_ids:
        x = np.asarray(data[subj][:, :epoch_len], dtype=np.float32)
        y = np.asarray(labels[subj], dtype=np.int64)
        # Model / Dataset expect (n, C, T) for CNN scorers.
        epochs = x[:, np.newaxis, :]
        if use_train_pool or epoch_holdout <= 0.0:
            eval_idx = np.arange(len(y))
        elif eval_indices is not None:
            eval_idx = _eval_idx_from_artifact(eval_indices, subj, len(y))
        else:
            _, eval_idx = stratified_epoch_holdout(
                y,
                epoch_holdout=epoch_holdout,
                seed=subject_seed(seed, subj),
                stratify=stratify,
            )
        subj_sels = simulate_samara_selections(
            epochs,
            y,
            eval_idx,
            phrase=phrase,
            r_max=r_max,
            seed=subject_seed(sim_seed, subj),
            grid=SAMARA_GRID,
        )
        for sel in subj_sels:
            sel.meta = {
                **sel.meta,
                "subject": subj,
                "label_source": "simulated",
                "data_path": str(data_path),
                "use_train_pool": use_train_pool,
                "eval_source": eval_source,
            }
            selections.append(sel)
    return selections


def merge_protocol_params(
    binary_cfg: dict[str, Any] | None,
    protocol_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Overlay speller ``protocol_params`` on binary experiment data params."""
    merged: dict[str, Any] = {}
    if binary_cfg:
        data = binary_cfg.get("data") or {}
        params = data.get("params") or {}
        if isinstance(params, dict):
            merged.update(params)
        # Promote common pipeline identity fields.
        if "pipeline" in data:
            merged.setdefault("pipeline", data["pipeline"])
    if protocol_params:
        merged.update(protocol_params)
    return merged
