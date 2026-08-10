"""Training loops for ContextualTransformer and SequenceClassifier."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from pattern_recognition.models.cnn import ContextualTransformer, SequenceClassifier
from pattern_recognition.training.metrics import compute_itr


def _to_device(
    batch: dict[str, torch.Tensor], device: str | torch.device
) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _estimate_pos_weight(dataloader, device: str | torch.device) -> torch.Tensor | None:
    """Estimate BCE ``pos_weight`` from train flash targets (valid flashes only)."""
    n_pos = 0
    n_neg = 0
    for batch in dataloader:
        targets = batch["flash_targets"]
        mask = batch["valid_mask"]
        n_pos += int((targets[mask] > 0.5).sum().item())
        n_neg += int((targets[mask] <= 0.5).sum().item())
    if n_pos == 0:
        return None
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)


def _binary_flash_metrics(
    preds: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float, float, float]:
    """Return accuracy, balanced accuracy, F1, ITR for binary flash predictions."""
    if preds.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0
    correct = (preds == targets).float().sum().item()
    acc = correct / preds.numel()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    bal_acc = 0.5 * (recall + specificity)
    f1 = (
        (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    )
    return acc, bal_acc, f1, compute_itr(acc, n_classes=2)


def selection_classifier_n_classes(model: SequenceClassifier) -> int:
    """ITR class count from SC head geometry (row×col joint, or cell ``n_cells``)."""
    if model.head_mode == "rowcol":
        return int(
            model.row_classifier.out_features * model.column_classifier.out_features
        )
    return int(model.character_classifier.out_features)


def train_contextual_transformer(
    model: ContextualTransformer,
    dataloaders: dict[str, Any],
    learning_params: dict[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """Train CT with masked BCE; metrics are flash-level on the valid mask.

    Returns ``(val_loss_history, metrics_dict, time_elapsed)`` with keys
    compatible with the binary runner writers (``Accuracy``, ``Balanced Accuracy``,
    ``F1-score``, ``ITR``).
    """
    since = time.time()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_params["lr"],
        weight_decay=learning_params["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=learning_params["step_size"],
        gamma=learning_params["gamma"],
    )
    model = model.to(device)
    pos_weight = _estimate_pos_weight(dataloaders["train"], device)

    val_loss_history: list[float] = []
    val_acc_history: list[float] = []
    val_bc_history: list[float] = []
    val_f1_history: list[float] = []
    val_itr_history: list[float] = []

    for _epoch in range(learning_params["num_epochs"]):
        for phase in ("train", "val"):
            model.train(phase == "train")
            running_loss = 0.0
            n_examples = 0
            all_preds: list[torch.Tensor] = []
            all_targets: list[torch.Tensor] = []

            for raw in dataloaders[phase]:
                batch = _to_device(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    flash_logits, valid_mask = model(
                        epochs=batch["epochs"],
                        stimulus_codes=batch["stimulus_codes"],
                        repetitions=batch["repetitions"],
                        valid_mask=batch["valid_mask"],
                    )
                    loss = ContextualTransformer.loss(
                        flash_logits,
                        batch["flash_targets"],
                        valid_mask,
                        pos_weight=pos_weight,
                    )
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                n_valid = int(valid_mask.sum().item())
                running_loss += float(loss.detach()) * max(n_valid, 1)
                n_examples += max(n_valid, 1)
                preds = (flash_logits[valid_mask] > 0).long().detach().cpu()
                targets = batch["flash_targets"][valid_mask].long().detach().cpu()
                all_preds.append(preds)
                all_targets.append(targets)

            epoch_loss = running_loss / max(n_examples, 1)
            preds_cat = (
                torch.cat(all_preds) if all_preds else torch.zeros(0, dtype=torch.long)
            )
            targets_cat = (
                torch.cat(all_targets)
                if all_targets
                else torch.zeros(0, dtype=torch.long)
            )
            acc, bal_acc, f1, itr = _binary_flash_metrics(preds_cat, targets_cat)

            if phase == "val":
                val_loss_history.append(epoch_loss)
                val_acc_history.append(acc)
                val_bc_history.append(bal_acc)
                val_f1_history.append(f1)
                val_itr_history.append(itr)

        scheduler.step()

    time_elapsed = time.time() - since
    metrics = {
        "Accuracy": np.array(val_acc_history),
        "Balanced Accuracy": np.array(val_bc_history),
        "F1-score": np.array(val_f1_history),
        "ITR": np.array(val_itr_history),
    }
    return np.array(val_loss_history), metrics, time_elapsed


def train_sequence_classifier(
    model: SequenceClassifier,
    dataloaders: dict[str, Any],
    learning_params: dict[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """Train SC with protocol CE heads.

    Primary ``Accuracy`` is row×col joint correctness (``head_mode='rowcol'``) or
    cell/character correctness (``head_mode='cell'``). ``Balanced Accuracy`` and
    ``F1-score`` are filled with NaN (selection accuracy is not a flash-level
    binary metric). ``ITR`` is computed from selection accuracy with the
    protocol class count.
    """
    since = time.time()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_params["lr"],
        weight_decay=learning_params["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=learning_params["step_size"],
        gamma=learning_params["gamma"],
    )
    model = model.to(device)

    val_loss_history: list[float] = []
    val_acc_history: list[float] = []
    n_classes = selection_classifier_n_classes(model)

    for _epoch in range(learning_params["num_epochs"]):
        for phase in ("train", "val"):
            model.train(phase == "train")
            running_loss = 0.0
            n_examples = 0
            n_correct = 0

            for raw in dataloaders[phase]:
                batch = _to_device(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    output = model(
                        epochs=batch["epochs"],
                        stimulus_codes=batch["stimulus_codes"],
                        repetitions=batch["repetitions"],
                        valid_mask=batch["valid_mask"],
                    )
                    loss = SequenceClassifier.loss(
                        output,
                        row_targets=batch.get("row_target"),
                        column_targets=batch.get("column_target"),
                        character_targets=batch.get("character_target"),
                    )
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                batch_size = batch["epochs"].size(0)
                running_loss += float(loss.detach()) * batch_size
                n_examples += batch_size

                if "row_logits" in output and "column_logits" in output:
                    row_ok = output["row_logits"].argmax(dim=-1) == batch["row_target"]
                    col_ok = (
                        output["column_logits"].argmax(dim=-1) == batch["column_target"]
                    )
                    n_correct += int((row_ok & col_ok).sum().item())
                else:
                    pred = output["character_logits"].argmax(dim=-1)
                    n_correct += int((pred == batch["character_target"]).sum().item())

            epoch_loss = running_loss / max(n_examples, 1)
            acc = n_correct / max(n_examples, 1)
            if phase == "val":
                val_loss_history.append(epoch_loss)
                val_acc_history.append(acc)

        scheduler.step()

    time_elapsed = time.time() - since
    acc_arr = np.array(val_acc_history, dtype=float)
    # Selection accuracy maps into Accuracy only; bal-acc / F1 are not defined.
    nan_arr = np.full_like(acc_arr, np.nan, dtype=float)
    itr_arr = np.array(
        [compute_itr(float(a), n_classes=n_classes) for a in acc_arr], dtype=float
    )
    metrics = {
        "Accuracy": acc_arr,
        "Balanced Accuracy": nan_arr,
        "F1-score": nan_arr.copy(),
        "ITR": itr_arr,
    }
    return np.array(val_loss_history), metrics, time_elapsed
