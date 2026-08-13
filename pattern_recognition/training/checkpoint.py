"""Keep the weights from the best validation epoch, not merely the last one."""

from __future__ import annotations

import copy
import math
from typing import Literal

import torch
from torch import nn

CheckpointMetric = Literal[
    "balanced_accuracy", "accuracy", "f1", "brier", "val_loss", "last"
]

#: Metrics where a smaller value is better.
_LOWER_IS_BETTER = frozenset({"val_loss", "brier"})

#: What each training loop can actually rank epochs by. ``SequenceClassifier``
#: reports selection accuracy only — balanced accuracy and F1 are NaN there
#: because a character decode is not a flash-level binary decision. Brier is
#: defined for both: flash-level for binary/CT, symbol-level for SC.
BINARY_METRICS = frozenset({"balanced_accuracy", "accuracy", "f1", "brier", "val_loss"})
SELECTION_METRICS = frozenset({"accuracy", "brier", "val_loss"})


def resolve_checkpoint_metric(requested: str, available: frozenset[str]) -> str:
    """Pick the closest usable ranking metric for a training loop.

    Falls back to accuracy when the requested metric is undefined for this
    model (rather than silently ranking every epoch as equally NaN). The
    resolved choice is recorded in ``run_meta.json`` so it is never a surprise.
    """
    if requested == "last" or requested in available:
        return requested
    if "accuracy" in available:
        return "accuracy"
    return "last"


class BestCheckpoint:
    """Track the best validation epoch and restore its weights at the end.

    Passed into a training loop by the caller; the loop calls :meth:`update`
    once per validation epoch and :meth:`restore` when training finishes. The
    caller then reads :attr:`best_epoch` to report the matching metrics.

    Note that using validation data to *select* an epoch makes those validation
    metrics mildly optimistic. The speller benchmark is unaffected: it scores a
    holdout that training never sees.
    """

    def __init__(self, metric: str = "brier") -> None:
        self.requested_metric = metric
        self.metric = metric
        self.best_epoch = 0
        self._best_value: float | None = None
        self._state: dict[str, torch.Tensor] | None = None
        self._last_epoch = 0

    def for_loop(self, available: frozenset[str]) -> BestCheckpoint:
        """Resolve the metric against what a specific loop can report."""
        self.metric = resolve_checkpoint_metric(self.requested_metric, available)
        return self

    @property
    def enabled(self) -> bool:
        return self.metric != "last"

    def update(self, epoch: int, value: float | None, model: nn.Module) -> None:
        self._last_epoch = epoch
        if not self.enabled or value is None:
            return
        value = float(value)
        if not math.isfinite(value):
            return
        if self._best_value is not None:
            # Ties go to the later epoch: a metric can sit on a plateau for
            # many epochs (balanced accuracy pins to exactly 0.5 while a P300
            # model still predicts all-negative), and keeping the first tie
            # would save the least-trained weights.
            improved = (
                value <= self._best_value
                if self.metric in _LOWER_IS_BETTER
                else value >= self._best_value
            )
            if not improved:
                return
        self._best_value = value
        self.best_epoch = epoch
        self._state = copy.deepcopy(
            {k: v.detach().cpu() for k, v in model.state_dict().items()}
        )

    def restore(self, model: nn.Module) -> int:
        """Load the best weights back into ``model``; return the chosen epoch."""
        if self._state is None:
            # Never improved (metric unavailable or zero epochs): keep as-is.
            self.best_epoch = self._last_epoch
            return self.best_epoch
        model.load_state_dict(self._state)
        return self.best_epoch
