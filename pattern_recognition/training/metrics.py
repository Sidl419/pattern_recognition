import numpy as np


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
