from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pattern_recognition.speller.grids import GridSpec
from pattern_recognition.speller.online import DecodeMode, online_decode
from pattern_recognition.speller.types import Selection
from pattern_recognition.training.metrics import compute_itr


def selection_duration_s(n_flashes_used: int, soa_s: float) -> float:
    return float(n_flashes_used * soa_s)


def _itr_bits_per_min(char_acc: float, n_classes: int, duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return compute_itr(char_acc, n_classes) * (60.0 / duration_s)


def evaluate_selections(
    selections: list[Selection],
    scores_per_sel: list[np.ndarray],
    decode_fn: Callable[[np.ndarray, Selection, int], str],
    repetitions: list[int],
    n_classes: int,
    soa_s: float,
    *,
    flashes_per_repeat: int,
    mode: DecodeMode,
    grid: GridSpec,
    early_stop: bool = False,
    margin_tau: float | None = None,
) -> dict:
    predictions: list[dict] = []
    correct_at_r: dict[int, list[bool]] = {r: [] for r in repetitions}

    for selection_id, (selection, scores) in enumerate(
        zip(selections, scores_per_sel, strict=True)
    ):
        subject = selection.meta.get("subject")
        for r in repetitions:
            pred = decode_fn(scores, selection, r)
            is_correct = pred == selection.target_char
            correct_at_r[r].append(is_correct)
            row = {
                "selection_id": selection_id,
                "r": r,
                "true": selection.target_char,
                "pred": pred,
            }
            if subject is not None:
                row["subject"] = subject
            predictions.append(row)

    acc_vs_repeats = []
    for r in repetitions:
        char_acc = float(np.mean(correct_at_r[r])) if correct_at_r[r] else 0.0
        duration_s = selection_duration_s(flashes_per_repeat * r, soa_s)
        itr = _itr_bits_per_min(char_acc, n_classes, duration_s)
        acc_vs_repeats.append({"r": r, "char_acc": char_acc, "itr": itr})

    result: dict = {
        "acc_vs_repeats": acc_vs_repeats,
        "predictions": predictions,
    }

    if early_stop:
        if margin_tau is None:
            raise ValueError("margin_tau is required when early_stop=True")
        r_max = max(repetitions)
        early_correct: list[bool] = []
        repeats_used: list[int] = []
        for selection, scores in zip(selections, scores_per_sel, strict=True):
            steps = online_decode(
                selection,
                scores,
                decode_fn,
                r_max=r_max,
                early_stop=True,
                margin_tau=margin_tau,
                mode=mode,
                grid=grid,
            )
            final_step = steps[-1]
            early_correct.append(final_step["pred"] == selection.target_char)
            repeats_used.append(final_step["r"])

        result["early_stop"] = {
            "char_acc_early": float(np.mean(early_correct)) if early_correct else 0.0,
            "mean_repeats_used": float(np.mean(repeats_used)) if repeats_used else 0.0,
            "repeats_used": [int(r) for r in repeats_used],
        }

    return result
