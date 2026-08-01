"""PICCL-only reliable-positive variant of DCCL's SupCon loss.

The original ``SupConLoss`` in ``algorithms.py`` is intentionally untouched.
With an all-one weight matrix this implementation is numerically equivalent to
the standard DCCL supervised contrastive objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReliableSupConLoss(nn.Module):
    def __init__(self, temperature=0.3):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = temperature

    def forward(self, features, labels, positive_weights):
        if len(features.shape) < 3:
            raise ValueError("features must have shape [B,V,...]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("labels must match the feature batch size")

        mask = torch.eq(labels, labels.transpose(0, 1)).float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.transpose(0, 1)),
            self.temperature,
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        if positive_weights.shape != mask.shape:
            raise ValueError(
                "positive_weights must match the expanded [B*V,B*V] SupCon mask"
            )
        weighted_positive_mask = mask * positive_weights.to(
            device=device, dtype=mask.dtype
        )
        denominator = weighted_positive_mask.sum(dim=1)
        numerator = (weighted_positive_mask * log_prob).sum(dim=1)
        mean_log_prob_pos = numerator / denominator.clamp_min(1e-12)
        mean_log_prob_pos = torch.where(
            denominator > 0, mean_log_prob_pos, torch.zeros_like(mean_log_prob_pos)
        )

        loss = -(
            self.temperature / self.base_temperature
        ) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()
