"""Load real EEG into speller ``Selection`` objects (BCI3 ground-truth, Samara sim)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import scipy.io as sio

from pattern_recognition.data.averaging import standardize_per_sample
from pattern_recognition.data.pipelines.bci3_pz import (
    DEFAULT_TEST_A_CHARS,
    DEFAULT_TEST_B_CHARS,
)
from pattern_recognition.data.p300 import P300Getter
from pattern_recognition.data.time_shift import load_p300_subjects
from pattern_recognition.speller.grids import SAMARA_GRID
from pattern_recognition.speller.simulate import (
    simulate_samara_selections,
    stratified_epoch_holdout,
)
from pattern_recognition.speller.types import Selection


def _flash_onsets(flashing: np.ndarray) -> np.ndarray:
    flash = np.asarray(flashing).ravel()
    idx = np.where(flash[:-1] != flash[1:])[0][1::2] + 1
    return np.concatenate([[0], idx])


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
    eloc_path = Path(eloc_path).expanduser().resolve()
    if not mat_path.is_file():
        raise FileNotFoundError(f"BCI3 .mat not found: {mat_path}")
    if not eloc_path.is_file():
        raise FileNotFoundError(f"Electrode montage not found: {eloc_path}")

    raw = sio.loadmat(str(mat_path))
    eloc = mne.channels.read_custom_montage(str(eloc_path))
    if len(eloc.ch_names) != n_channels:
        raise ValueError(
            f"Montage has {len(eloc.ch_names)} channels but n_channels={n_channels}"
        )

    target_chars = _resolve_bci3_target_chars(raw, subject, test_chars)
    n_chars = len(raw["Flashing"])
    if max_chars is not None:
        n_chars = min(n_chars, int(max_chars))
    if len(target_chars) < n_chars:
        raise ValueError(
            f"Only {len(target_chars)} target chars for {n_chars} character epochs"
        )

    getter = P300Getter(
        raw,
        eloc,
        n_channels=n_channels,
        sfreq=sfreq,
        sample_size=sample_size,
        target_chars=target_chars,
    )
    names = [c.lower().strip(".") for c in eloc.ch_names]
    ch_key = channel_name.lower().strip(".")
    try:
        ch_idx = names.index(ch_key)
    except ValueError as exc:
        raise KeyError(
            f"Channel {channel_name!r} not in montage {eloc.ch_names}"
        ) from exc

    selections: list[Selection] = []
    for epoch_num in range(n_chars):
        idx = _flash_onsets(raw["Flashing"][epoch_num])
        if apply_filter:
            data = getter.filter(raw["Signal"][epoch_num])
        else:
            data = raw["Signal"][epoch_num].T

        flashes = []
        for onset in idx:
            flashes.append(data[ch_idx, onset : onset + sample_size])
        flashes_arr = np.stack(flashes, axis=0).astype(np.float32)
        flashes_arr = standardize_per_sample(flashes_arr)
        flashes_arr = flashes_arr[:, np.newaxis, :]

        codes = np.asarray(raw["StimulusCode"][epoch_num][idx], dtype=np.int64)
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
                },
            )
        )
    return selections


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
) -> list[Selection]:
    """Load Samara epochs and simulate single-flash selections on holdout pools."""
    data_path = Path(data_path).expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(f"Samara data path not found: {data_path}")

    data, labels = load_p300_subjects(
        str(data_path),
        channel_idx=channel_idx,
        standardize=True,
        file_pattern=file_pattern,
    )
    if not data:
        raise FileNotFoundError(
            f"No files matching {file_pattern!r} under {data_path}"
        )

    subject_ids = sorted(data.keys())
    if subjects is not None:
        missing = [s for s in subjects if s not in data]
        if missing:
            raise KeyError(f"Subjects {missing} not in loaded set {subject_ids}")
        subject_ids = list(subjects)

    sim_seed = seed if simulation_seed is None else simulation_seed
    selections: list[Selection] = []
    for i, subj in enumerate(subject_ids):
        x = np.asarray(data[subj][:, :epoch_len], dtype=np.float32)
        y = np.asarray(labels[subj], dtype=np.int64)
        # Model / Dataset expect (n, C, T) for CNN scorers.
        epochs = x[:, np.newaxis, :]
        _, eval_idx = stratified_epoch_holdout(
            y,
            epoch_holdout=epoch_holdout,
            seed=seed + i,
            stratify=stratify,
        )
        subj_sels = simulate_samara_selections(
            epochs,
            y,
            eval_idx,
            phrase=phrase,
            r_max=r_max,
            seed=sim_seed + i,
            grid=SAMARA_GRID,
        )
        for sel in subj_sels:
            sel.meta = {
                **sel.meta,
                "subject": subj,
                "label_source": "simulated",
                "data_path": str(data_path),
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
