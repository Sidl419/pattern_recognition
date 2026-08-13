import torch
from torch import nn
from torch.nn import functional as F


class BrierLoss(nn.Module):
    """Multi-category Brier score as a training objective.

    Brier is a proper scoring rule that stays bounded under heavy class
    imbalance, which is why this codebase prefers it to cross-entropy on
    ~1:15 P300 data.

    The loss owns the normalisation: it applies the softmax itself and expects
    **logits**, so a model cannot silently opt out of being scored properly by
    forgetting its output activation.

    ``reduction="mean"`` averages over samples *and* classes, i.e. the standard
    Brier score divided by ``n_classes``. That matches the scale of the plain
    ``nn.MSELoss()`` this replaces, so existing learning rates carry over
    unchanged. Use :func:`pattern_recognition.training.metrics.brier_score`
    for the reported metric, which uses the conventional summed-over-classes
    definition.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        """Class probabilities matching the loss's own link function."""
        return logits.softmax(dim=1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """``logits`` ``[N, K]`` against one-hot ``targets`` ``[N, K]``."""
        if logits.shape != targets.shape:
            raise ValueError(
                f"BrierLoss expects matching shapes, got logits {tuple(logits.shape)} "
                f"and targets {tuple(targets.shape)} (targets must be one-hot)"
            )
        return F.mse_loss(
            self.probabilities(logits), targets.float(), reduction=self.reduction
        )


class GraphLoss(nn.Module):
    def __init__(self, base, alpha=1, beta=1, gamma=1):
        super(GraphLoss, self).__init__()
        self.base = base
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, predictios, adj_matrix, labels, inputs):
        predictios = predictios.view(-1)
        labels = labels.view(-1)
        n = adj_matrix.size(2)

        base_loss = self.base(predictios, labels)  # F.mse_loss

        # degree_matrix = adj_matrix.sum(dim=1)
        # laplacian = degree_matrix - adj_matrix
        # laplacian
        smoothness_term = (
            torch.mean(torch.cdist(inputs, inputs, p=2).pow(2) * adj_matrix, dim=(1, 2))
            / 2
        )

        connectivity_term = -torch.log(adj_matrix.sum(dim=1)).mean(dim=1)

        sparsity_term = torch.norm(adj_matrix, dim=(1, 2)) / (n * n)

        graph_loss = (
            self.alpha * smoothness_term
            + self.beta * connectivity_term
            + self.gamma * sparsity_term
        )

        return base_loss + graph_loss.mean()
