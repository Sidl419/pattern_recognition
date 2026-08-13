"""Shared BCI3 flash extraction matching ``BCI3PzEpochAverage`` preprocessing."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import scipy.io as sio

from pattern_recognition.data.averaging import standardize_per_sample
from pattern_recognition.data.p300 import P300Getter

# Published target strings for the BCI III test sets (labels hidden in the
# .mat files). Kept here rather than in a pipeline module so both the epoch
# pipeline and selection loading can use them without importing each other.
DEFAULT_TEST_A_CHARS = list(
    "WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU"
)
DEFAULT_TEST_B_CHARS = list(
    "MERMIROOMUHJPXJOHUVLEORWRJBTJGGCIFSCGGLFSLEZWNPMPKEZSYSENDWPBDUUJIXZETUZCWOJQQRTVXBMZWHSWPRQDZEQRCOQZ"
)


def _channel_index(eloc, channel_name: str) -> int:
    names = [c.lower().strip(".") for c in eloc.ch_names]
    key = channel_name.lower().strip(".")
    try:
        return names.index(key)
    except ValueError as exc:
        raise KeyError(
            f"Channel {channel_name!r} not in montage {eloc.ch_names}"
        ) from exc


def _flash_onsets(flashing: np.ndarray) -> np.ndarray:
    flash = np.asarray(flashing).ravel()
    idx = np.where(flash[:-1] != flash[1:])[0][1::2] + 1
    return np.concatenate([[0], idx])


def prepare_bci3_pz_flashes(
    mat_path: str | Path,
    *,
    eloc_path: str | Path,
    channel_name: str = "Pz",
    n_channels: int = 64,
    sfreq: float = 120.0,
    sample_size: int = 72,
    apply_filter: bool = True,
    target_chars: list[str] | None = None,
    max_chars: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Extract per-character Pz flashes with the binary-pipeline preprocessing.

    Mirrors ``P300Getter.get_cnn_p300_dataset`` + ``get_data`` (MNE ``Scaler``
    fit on all-channel flashes), then selects ``channel_name`` and applies
    ``standardize_per_sample``.

    Returns
    -------
    flashes_list
        Per character, array shaped ``(n_flashes, sample_size)``.
    codes_list
        Per character, stimulus codes aligned with flashes.
    onsets_list
        Per character, flash onset sample indices in the continuous signal.
    """
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

    getter = P300Getter(
        raw,
        eloc,
        n_channels=n_channels,
        sfreq=sfreq,
        sample_size=sample_size,
        target_chars=target_chars,
    )
    ch_idx = _channel_index(eloc, channel_name)

    n_chars = len(raw["Flashing"])
    if max_chars is not None:
        n_chars = min(n_chars, int(max_chars))
    if target_chars is not None and len(target_chars) < n_chars:
        raise ValueError(
            f"Only {len(target_chars)} target chars for {n_chars} character epochs"
        )

    per_char_counts: list[int] = []
    all_flashes: list[np.ndarray] = []
    codes_list: list[np.ndarray] = []
    onsets_list: list[np.ndarray] = []

    for epoch_num in range(n_chars):
        onsets = _flash_onsets(raw["Flashing"][epoch_num])
        if apply_filter:
            data = getter.filter(raw["Signal"][epoch_num])
        else:
            data = raw["Signal"][epoch_num].T

        flashes = []
        for onset in onsets:
            flashes.append(data[:, onset : onset + sample_size])
        flashes_arr = np.stack(flashes, axis=0).astype(np.float64)
        codes = np.asarray(raw["StimulusCode"][epoch_num][onsets], dtype=np.int64)

        per_char_counts.append(len(flashes_arr))
        all_flashes.append(flashes_arr)
        codes_list.append(codes)
        onsets_list.append(onsets.astype(np.int64))

    stacked = np.concatenate(all_flashes, axis=0)
    scaled = getter.scaler.fit_transform(stacked)
    scaled = np.asarray(scaled, dtype=np.float32)

    flashes_list: list[np.ndarray] = []
    offset = 0
    for count in per_char_counts:
        block = scaled[offset : offset + count, ch_idx, :]
        offset += count
        flashes_list.append(standardize_per_sample(block))

    return flashes_list, codes_list, onsets_list
