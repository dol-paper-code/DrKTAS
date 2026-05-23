"""Stage 1 loss functions.

The Stage 1 objective combines a generation loss and a classification loss
on a shared backbone representation (equation (6) in the paper):

``L_total = L_gen + alpha * L_cls``,    with ``alpha = 0.5``.

The classification term (equations (3) and (4)) is the weighted
cross-entropy with class-prior weights computed by inverse-frequency
reweighting, augmented by an expected-ordinal-distance penalty:

``L_cls = L_WCE(z, y) + lambda_ord * sum_k p_k * |k - y| / 4``,
``lambda_ord = 0.5``.

The combiner also exposes the Kendall-style uncertainty weighting alternative
for sensitivity studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedOrdinalCrossEntropy(nn.Module):
    """Weighted cross-entropy with an expected-ordinal-distance penalty.

    Parameters
    ----------
    num_classes:
        Number of ordinal classes (default ``5`` for KTAS levels 1-5).
    class_weights:
        Sequence of length ``num_classes`` giving the WCE weight ``w_y``
        for each class. When ``None``, all weights are 1 and the term
        reduces to standard cross-entropy.
    lambda_ord:
        Coefficient on the expected ordinal-distance penalty. The default
        ``0.5`` reproduces the value used in the paper.
    """

    def __init__(
        self,
        num_classes: int = 5,
        class_weights: Optional[Sequence[float]] = None,
        lambda_ord: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_ord = float(lambda_ord)

        if class_weights is None:
            weights = torch.ones(num_classes, dtype=torch.float32)
        else:
            weights = torch.tensor(list(class_weights), dtype=torch.float32)
            if weights.numel() != num_classes:
                raise ValueError(
                    f"class_weights has length {weights.numel()} but "
                    f"num_classes={num_classes}."
                )
        self.register_buffer("class_weights", weights)

        # Pairwise normalized distance |i - j| / (K - 1). This is the
        # |k - y| / 4 factor in the paper when K = 5.
        distance = torch.zeros(num_classes, num_classes, dtype=torch.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                distance[i, j] = abs(i - j) / (num_classes - 1)
        self.register_buffer("distance_matrix", distance)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute the weighted CE + ordinal-penalty loss.

        Parameters
        ----------
        logits:
            Float tensor of shape ``(batch, num_classes)``.
        targets:
            Long tensor of shape ``(batch,)`` with values in
            ``[0, num_classes - 1]``.
        reduction:
            ``'mean'``, ``'sum'`` or ``'none'``.
        """
        weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        ce = F.cross_entropy(logits, targets, weight=weights, reduction="none")

        probs = F.softmax(logits, dim=-1)
        per_class_distance = self.distance_matrix.to(device=logits.device, dtype=probs.dtype)
        expected_distance = (probs * per_class_distance[targets]).sum(dim=-1)

        total = ce + self.lambda_ord * expected_distance
        if reduction == "mean":
            return total.mean()
        if reduction == "sum":
            return total.sum()
        return total

    @torch.no_grad()
    def mean_predicted_distance(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Mean ``|argmax(logits) - targets|`` for diagnostics."""
        preds = logits.argmax(dim=-1)
        return torch.abs(preds.float() - targets.float()).mean().item()


class UncertaintyWeighting(nn.Module):
    """Homoscedastic uncertainty weighting for multi-task losses.

    Implements the formulation in Kendall, Gal & Cipolla (CVPR 2018):

    ``L = sum_i 0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i``.
    """

    def __init__(self, num_tasks: int = 2) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self, losses: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if len(losses) != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} losses, received {len(losses)}."
            )
        total = losses[0].new_zeros(())
        info: Dict[str, float] = {}
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + 0.5 * precision * loss + 0.5 * self.log_vars[i]
            info[f"task_{i}_weight"] = precision.item()
            info[f"task_{i}_sigma"] = torch.exp(0.5 * self.log_vars[i]).item()
        return total, info

    @torch.no_grad()
    def effective_weights(self) -> List[float]:
        return [torch.exp(-lv).item() for lv in self.log_vars]


@dataclass
class LossCombiner:
    """Combine the generation and classification losses for Stage 1.

    Two strategies are supported:

    * ``'simple_sum'``: ``L = L_gen + cls_weight * L_cls`` (the paper default
      with ``cls_weight = 0.5``).
    * ``'uncertainty_weighting'``: Kendall et al. (2018) homoscedastic
      uncertainty weighting, exposed for sensitivity studies.
    """

    method: str = "simple_sum"
    cls_weight: float = 0.5
    uncertainty_weighting: Optional[UncertaintyWeighting] = None

    def __post_init__(self) -> None:
        if self.method not in ("simple_sum", "uncertainty_weighting"):
            raise ValueError(f"Unknown loss combination method: {self.method}")
        if self.method == "uncertainty_weighting" and self.uncertainty_weighting is None:
            raise ValueError(
                "method='uncertainty_weighting' requires an UncertaintyWeighting module."
            )

    def combine(
        self,
        gen_loss: torch.Tensor,
        cls_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.method == "simple_sum":
            combined = gen_loss + self.cls_weight * cls_loss
            info = {"gen_weight": 1.0, "cls_weight": float(self.cls_weight)}
            return combined, info

        assert self.uncertainty_weighting is not None
        return self.uncertainty_weighting([gen_loss, cls_loss])
