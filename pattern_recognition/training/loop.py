import time

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint
from torch import optim
from torch.optim.lr_scheduler import StepLR

from pattern_recognition.losses import BrierLoss, GraphLoss
from pattern_recognition.training.checkpoint import BINARY_METRICS
from pattern_recognition.training.metrics import brier_score, compute_itr


def validate_model(model, dataloader, is_binary=True, device="cpu"):
    model = model.to(device)
    model.eval()

    running_corrects = 0
    all_probs, all_targets = [], []
    if is_binary:
        running_TP, running_TN, running_FP, running_FN = 0, 0, 0, 0

    for data in dataloader:
        inputs = data[0].to(device)
        labels = data[1].to(device)
        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        _, true_y = torch.max(labels.data, 1)
        all_probs.append(BrierLoss.probabilities(outputs).detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

        if is_binary:
            P = torch.sum(preds)
            N = torch.sum(1 - preds)
            TP = torch.sum(torch.masked_select(true_y, preds == 1))
            TN = torch.sum(torch.masked_select(1 - true_y, preds == 0))
            FP = P - TP
            FN = N - TN

        running_corrects += torch.sum(preds == true_y)
        if is_binary:
            running_TP += TP
            running_TN += TN
            running_FP += FP
            running_FN += FN

    acc = running_corrects.double() / len(dataloader.dataset)

    if is_binary:
        precision = (
            running_TP.double() / (running_TP + running_FP)
            if running_TP + running_FP != 0
            else torch.tensor(0)
        )
        recall = (
            running_TP.double() / (running_TP + running_FN)
            if running_TP + running_FN != 0
            else torch.tensor(0)
        )
        f1 = (
            (2 * (precision * recall) / (precision + recall))
            if precision + recall != 0
            else torch.tensor(0)
        )
        bc = (recall + running_TN.double() / (running_TN + running_FP)) / 2

    min_acc, max_acc = proportion_confint(
        running_corrects.cpu(), len(dataloader.dataset), 0.05
    )
    acc_val = acc.cpu().item()
    acc = {
        "Accuracy": acc_val,
        "Corrects": running_corrects.cpu().item(),
        "Min Accuracy": min_acc,
        "Max Accuracy": max_acc,
        "Brier": brier_score(np.concatenate(all_probs), np.concatenate(all_targets)),
    }
    if is_binary:
        acc["Balanced Accuracy"] = bc
        acc["F1-score"] = f1.cpu().item()
        acc["ITR"] = compute_itr(acc_val, n_classes=2)
    return acc


def train_model(
    model,
    dataloaders,
    criterion,
    learning_params,
    is_binary=True,
    device="cpu",
    log_rate=None,
    val_rate=1,
    checkpoint=None,
):
    """Train a CNN/GNN model.

    ``checkpoint`` is an optional :class:`~pattern_recognition.training.
    checkpoint.BestCheckpoint`; when given, the model is left holding the
    weights from its best validation epoch instead of the last one.
    """
    since = time.time()
    if checkpoint is not None:
        checkpoint.for_loop(BINARY_METRICS if is_binary else frozenset({"val_loss"}))

    optimizer = optim.AdamW(
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

    (
        val_corrects_history,
        val_acc_history,
        val_loss_history,
        val_f1_history,
        val_bc_history,
    ) = [], [], [], [], []
    val_min_acc_history, val_max_acc_history = [], []
    val_itr_history = []
    val_brier_history = []

    for epoch in range(learning_params["num_epochs"]):
        do_val = (
            (val_rate <= 1)
            or ((epoch + 1) % val_rate == 0)
            or (epoch == learning_params["num_epochs"] - 1)
        )
        for phase in ["train", "val"] if do_val else ["train"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            epoch_probs: list[np.ndarray] = []
            epoch_targets: list[np.ndarray] = []
            if is_binary:
                running_ones = 0
                running_TP, running_TN, running_FP, running_FN = 0, 0, 0, 0

            for data in dataloaders[phase]:
                if learning_params["model_type"] == "GNN":
                    inputs = data.to(device)
                    labels = data.y.to(device)
                    inputs_size = inputs.x.size(0)
                elif learning_params["model_type"] == "CNN":
                    inputs = data[0].to(device)
                    labels = data[1].to(device)
                    inputs_size = inputs.size(0)
                else:
                    raise ValueError(
                        f"no such model type: {learning_params['model_type']}"
                    )

                optimizer.zero_grad()
                if isinstance(criterion, GraphLoss):
                    outputs, adj = model(inputs)
                    loss = criterion(outputs, adj, labels, inputs)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                _, true_y = torch.max(labels.data, 1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                if is_binary:
                    P = torch.sum(preds)
                    N = torch.sum(1 - preds)
                    TP = torch.sum(torch.masked_select(true_y, preds == 1))
                    TN = torch.sum(torch.masked_select(1 - true_y, preds == 0))
                    FP = P - TP
                    FN = N - TN

                running_loss += loss.item() * inputs_size
                running_corrects += torch.sum(preds == true_y)
                epoch_probs.append(
                    BrierLoss.probabilities(outputs).detach().cpu().numpy()
                )
                epoch_targets.append(labels.detach().cpu().numpy())

                if is_binary:
                    running_ones += P
                    running_TP += TP
                    running_TN += TN
                    running_FP += FP
                    running_FN += FN

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if is_binary:
                epoch_ones = running_ones.double() / (
                    len(dataloaders[phase].dataset) // dataloaders[phase].batch_size
                )
                epoch_precision = (
                    running_TP.double() / (running_TP + running_FP)
                    if running_TP + running_FP != 0
                    else torch.tensor(0)
                )
                epoch_recall = (
                    running_TP.double() / (running_TP + running_FN)
                    if running_TP + running_FN != 0
                    else torch.tensor(0)
                )
                epoch_f1 = (
                    (
                        2
                        * (epoch_precision * epoch_recall)
                        / (epoch_precision + epoch_recall)
                    )
                    if epoch_precision + epoch_recall != 0
                    else torch.tensor(0)
                )
                epoch_bc = (
                    epoch_recall + running_TN.double() / (running_TN + running_FP)
                ) / 2

            min_acc, max_acc = proportion_confint(
                running_corrects.cpu(), len(dataloaders[phase].dataset), 0.05
            )

            epoch_itr = (
                compute_itr(epoch_acc.cpu().item(), n_classes=2) if is_binary else 0.0
            )
            epoch_brier = brier_score(
                np.concatenate(epoch_probs), np.concatenate(epoch_targets)
            )

            if (log_rate is not None) and (epoch + 1) % log_rate == 0:
                if phase == "train":
                    print(
                        "Epoch {}/{}".format(epoch, learning_params["num_epochs"] - 1)
                    )
                    print("-" * 150)
                if is_binary:
                    print(
                        "{}\t Loss: {:.4f}\t Min Acc: {:.4f}\t Acc: {:.4f}\t Max Acc: {:.4f}\t Balanced Acc: {:.4f}\t Positive: {:.4f}\t Precision: {:.4f}\t Recall: {:.4f}\t ITR: {:.4f}\t".format(
                            phase,
                            epoch_loss,
                            min_acc,
                            epoch_acc,
                            max_acc,
                            epoch_bc,
                            epoch_ones,
                            epoch_precision,
                            epoch_recall,
                            epoch_itr,
                        )
                    )
                else:
                    print(
                        "{}\t Loss: {:.4f}\t Min Acc: {:.4f}\t Acc: {:.4f}\t Max Acc: {:.4f}\t".format(
                            phase, epoch_loss, min_acc, epoch_acc, max_acc
                        )
                    )

            if phase == "val":
                val_acc_history.append(epoch_acc.cpu().data)
                val_corrects_history.append(running_corrects.cpu().data)
                val_loss_history.append(epoch_loss)
                val_min_acc_history.append(min_acc)
                val_max_acc_history.append(max_acc)
                val_brier_history.append(epoch_brier)
                if is_binary:
                    val_f1_history.append(epoch_f1.cpu())
                    val_bc_history.append(epoch_bc.cpu())
                    val_itr_history.append(epoch_itr)

                if checkpoint is not None:
                    ranked = {
                        "accuracy": float(epoch_acc),
                        "val_loss": float(epoch_loss),
                        "brier": epoch_brier,
                    }
                    if is_binary:
                        ranked["balanced_accuracy"] = float(epoch_bc)
                        ranked["f1"] = float(epoch_f1)
                    checkpoint.update(
                        len(val_acc_history) - 1, ranked.get(checkpoint.metric), model
                    )

        scheduler.step()

    if checkpoint is not None:
        checkpoint.restore(model)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    acc = {
        "Accuracy": np.array(val_acc_history),
        "Corrects": np.array(val_corrects_history),
        "Min Accuracy": np.array(val_min_acc_history),
        "Max Accuracy": np.array(val_max_acc_history),
        "Brier": np.array(val_brier_history),
    }
    if is_binary:
        acc["Balanced Accuracy"] = np.array(val_bc_history)
        acc["F1-score"] = np.array(val_f1_history)
        acc["ITR"] = np.array(val_itr_history)

    return np.array(val_loss_history), acc, time_elapsed


def plot_sample(raw_dataset, signal_sample, info, is_mean=False):
    output = raw_dataset.unscale(signal_sample.numpy())[0]

    plt.figure(figsize=(10, 10))
    mean_output = output.mean(axis=0)
    t_axis = np.arange(len(mean_output)) / info["sfreq"] * 1000
    plt.plot(t_axis, mean_output)
    plt.ylabel("amplitude (muV)")
    plt.xlabel("time (ms)")
    plt.title("Averaged EEG signal")
    plt.show()

    mne_output = mne.io.RawArray(output, info=info, verbose=False)
    plt.figure(figsize=(10, 10))
    mne_output.plot(
        n_channels=len(info["ch_names"]), scalings="auto", title="Raw EEG signal"
    )
    plt.show()


def show_progress(loss, metric, loss_title, metric_title):
    plt.figure(figsize=(10, 6))
    epochs = np.arange(len(loss))

    plt.plot(epochs, loss, "r-", linewidth=2, label=loss_title)
    plt.plot(epochs, metric[metric_title], "b-", linewidth=2, label=metric_title)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.title(f"{loss_title} & {metric_title} over Epochs", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()


def paired_proportions_exact_test(preds_a, preds_b, targets):
    preds_a = preds_a == targets
    preds_b = preds_b == targets

    a = sum((preds_a == 1) & (preds_b == 1))
    b = sum((preds_a == 1) & (preds_b == 0))
    c = sum((preds_a == 0) & (preds_b == 1))
    d = sum((preds_a == 0) & (preds_b == 0))
    print([[a, b], [c, d]])

    return mcnemar([[a, b], [c, d]], exact=True).pvalue


def infer_model(model, dataloader, channel=None, device="cpu", model_type="CNN"):
    model = model.to(device)
    model.eval()
    all_preds = []

    for data in dataloader:
        if model_type == "GNN":
            inputs = data.to(device)
        elif model_type == "CNN":
            inputs = data[0].to(device)
        else:
            raise ValueError(f"no such model type: {model_type}")

        with torch.no_grad():
            if channel is not None:
                inputs = inputs[:, channel].unsqueeze(1)
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)

        all_preds.append(outputs)
    return torch.cat(all_preds).cpu()
