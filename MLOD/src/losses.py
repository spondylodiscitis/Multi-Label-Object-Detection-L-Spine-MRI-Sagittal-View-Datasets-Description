from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_positive: float = 0.0,
        gamma_negative: float = 4.0,
        clip: float = 0.05,
        epsilon: float = 1e-8,
        class_counts: Optional[Sequence[int]] = None,
        class_balance_beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.gamma_positive = gamma_positive
        self.gamma_negative = gamma_negative
        self.clip = clip
        self.epsilon = epsilon

        if class_counts is None:
            self.register_buffer("class_weights", None)
        else:
            counts = torch.tensor(class_counts, dtype=torch.float32)
            total = counts.sum()
            weights = (total / (counts + 1e-6)).pow(class_balance_beta)
            self.register_buffer("class_weights", weights / weights.mean())

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        positive_probability = probabilities
        negative_probability = 1.0 - probabilities

        if self.clip and self.clip > 0:
            negative_probability = (
                negative_probability + self.clip
            ).clamp(max=1.0)

        positive_loss = targets * torch.log(
            positive_probability.clamp(min=self.epsilon)
        )
        negative_loss = (1.0 - targets) * torch.log(
            negative_probability.clamp(min=self.epsilon)
        )

        positive_weight = (
            (1.0 - positive_probability) ** self.gamma_positive
        ) * targets
        negative_weight = (
            (1.0 - negative_probability) ** self.gamma_negative
        ) * (1.0 - targets)

        loss = -(
            positive_loss * positive_weight
            + negative_loss * negative_weight
        )

        if self.class_weights is not None:
            loss = loss * self.class_weights.view(1, -1)

        return loss.mean()
