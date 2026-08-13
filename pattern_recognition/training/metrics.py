import numpy as np


def brier_score(probabilities, targets_onehot):
    """Multi-category Brier score: mean over samples of the summed squared error.

    ``BS = (1/N) * sum_i sum_k (p_ik - o_ik)^2`` — the conventional definition
    (Brier 1950), so a binary score is twice the single-component form and the
    range is ``[0, 2]``. Lower is better. Unlike accuracy it rewards calibrated
    probabilities rather than the arg-max alone, which is why it degrades more
    gracefully than accuracy under strong class imbalance.

    ``probabilities`` must already be normalised (e.g. via
    :meth:`~pattern_recognition.losses.BrierLoss.probabilities`).
    """
    p = np.asarray(probabilities, dtype=float)
    o = np.asarray(targets_onehot, dtype=float)
    if p.shape != o.shape:
        raise ValueError(
            f"brier_score shape mismatch: probabilities {p.shape} vs targets {o.shape}"
        )
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.sum((p - o) ** 2, axis=1)))


def compute_itr(accuracy, n_classes=2):
    """
    Wolpaw's Information Transfer Rate (bits/trial).
    ITR = log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))
    where N = number of classes, P = accuracy in [0, 1].
    """
    P = float(accuracy)
    N = int(n_classes)
    if P <= 0 or P >= 1:
        P = np.clip(P, 1e-10, 1 - 1e-10)
    if N <= 1:
        return 0.0
    itr = np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))
    return float(itr)
