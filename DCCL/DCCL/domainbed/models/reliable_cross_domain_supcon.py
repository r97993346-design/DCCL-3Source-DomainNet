"""Reliable cross-domain positive weighting for DCCL.

This module intentionally ignores augmented views when estimating reliability.
Only original-sample CBB features are used to maintain class/domain prototypes
and validate cross-domain same-class pairs. Augmented views merely inherit the
resulting original-sample pair weights inside supervised contrastive learning.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossDomainPrototypeReliability(nn.Module):
    """Build pair weights by bidirectional class/domain prototype validation.

    For a same-class pair ``(i, j)`` from different source domains, reliability
    is computed in both directions::

        r_i_to_j = (1 + cos(h_i, prototype[y_i, domain_j])) / 2
        r_j_to_i = (1 + cos(h_j, prototype[y_j, domain_i])) / 2
        rho_ij = sqrt(r_i_to_j * r_j_to_i)

    The resulting reliability is normalized by an EMA baseline and clipped
    around one. All non-cross-domain-positive entries remain one.
    """

    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        feature_dim: int,
        proto_momentum: float = 0.99,
        reliability_momentum: float = 0.99,
        start_step: int = 500,
        weight_min: float = 0.5,
        weight_max: float = 1.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_classes <= 0 or num_domains <= 0 or feature_dim <= 0:
            raise ValueError("num_classes, num_domains and feature_dim must be positive")
        if not 0.0 <= proto_momentum < 1.0:
            raise ValueError("proto_momentum must be in [0, 1)")
        if not 0.0 <= reliability_momentum < 1.0:
            raise ValueError("reliability_momentum must be in [0, 1)")
        if weight_min <= 0.0 or weight_max < weight_min:
            raise ValueError("Require 0 < weight_min <= weight_max")

        self.num_classes = int(num_classes)
        self.num_domains = int(num_domains)
        self.feature_dim = int(feature_dim)
        self.proto_momentum = float(proto_momentum)
        self.reliability_momentum = float(reliability_momentum)
        self.start_step = int(start_step)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.eps = float(eps)

        self.register_buffer(
            "prototypes",
            torch.zeros(self.num_classes, self.num_domains, self.feature_dim),
        )
        self.register_buffer(
            "prototype_valid",
            torch.zeros(self.num_classes, self.num_domains, dtype=torch.bool),
        )
        self.register_buffer("reliability_ema", torch.tensor(1.0))
        self.register_buffer(
            "reliability_ema_initialized",
            torch.tensor(False, dtype=torch.bool),
        )

        self.last_stats: Dict[str, float] = {
            "cross_pairs": 0.0,
            "valid_pairs": 0.0,
            "valid_pair_fraction": 0.0,
            "reliability_mean": 1.0,
            "weight_mean": 1.0,
            "weight_min": 1.0,
            "weight_max": 1.0,
        }

    @torch.no_grad()
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        domains: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Return a ``[B, B]`` original-sample pair-weight matrix.

        ``features`` must come only from original samples. The method computes
        weights using historical prototypes first and updates prototypes with
        the current batch afterwards, preventing samples from constructing
        their own reference prototype.
        """
        self._validate_inputs(features, labels, domains)

        features = F.normalize(features.detach(), dim=1)
        labels = labels.detach().long()
        domains = domains.detach().long()
        batch_size = features.shape[0]

        pair_weight = torch.ones(
            batch_size,
            batch_size,
            device=features.device,
            dtype=features.dtype,
        )

        same_class = labels[:, None].eq(labels[None, :])
        cross_domain = domains[:, None].ne(domains[None, :])
        cross_positive = same_class & cross_domain
        cross_pair_count = int(cross_positive.sum().item())

        # During warm-up, prototypes are populated but DCCL remains exactly
        # unweighted. No augmentation feature is used here or elsewhere.
        if int(step) < self.start_step:
            self._update_prototypes(features, labels, domains)
            self._update_stats(cross_pair_count=cross_pair_count)
            return pair_weight

        # Validate only actual cross-domain positive indices. This avoids
        # constructing a dense [B, B, F] tensor when F is 2048 for ResNet-50.
        pair_i, pair_j = cross_positive.nonzero(as_tuple=True)
        if pair_i.numel() > 0:
            valid_i_to_j = self.prototype_valid[labels[pair_i], domains[pair_j]]
            valid_j_to_i = self.prototype_valid[labels[pair_j], domains[pair_i]]
            pair_valid = valid_i_to_j & valid_j_to_i
            pair_i = pair_i[pair_valid]
            pair_j = pair_j[pair_valid]

        if pair_i.numel() > 0:
            proto_i_to_j = self.prototypes[labels[pair_i], domains[pair_j]]
            proto_j_to_i = self.prototypes[labels[pair_j], domains[pair_i]]

            rel_i_to_j = (
                1.0 + (features[pair_i] * proto_i_to_j).sum(dim=1)
            ) * 0.5
            rel_j_to_i = (
                1.0 + (features[pair_j] * proto_j_to_i).sum(dim=1)
            ) * 0.5

            valid_reliability = torch.sqrt(
                rel_i_to_j.clamp(0.0, 1.0).clamp_min(self.eps)
                * rel_j_to_i.clamp(0.0, 1.0).clamp_min(self.eps)
            )
            batch_reliability_mean = valid_reliability.mean()

            # The first valid active batch is normalized by its own mean;
            # subsequent batches use the historical EMA baseline.
            if bool(self.reliability_ema_initialized.item()):
                baseline = self.reliability_ema.clamp_min(self.eps)
            else:
                baseline = batch_reliability_mean.clamp_min(self.eps)

            valid_weights = (valid_reliability / baseline).clamp(
                min=self.weight_min,
                max=self.weight_max,
            )
            pair_weight[pair_i, pair_j] = valid_weights

            if bool(self.reliability_ema_initialized.item()):
                self.reliability_ema.mul_(self.reliability_momentum).add_(
                    batch_reliability_mean * (1.0 - self.reliability_momentum)
                )
            else:
                self.reliability_ema.copy_(batch_reliability_mean)
                self.reliability_ema_initialized.fill_(True)

            self._update_stats(
                cross_pair_count=cross_pair_count,
                valid_pair_count=int(pair_i.numel()),
                reliability_mean=float(batch_reliability_mean.item()),
                weight_mean=float(valid_weights.mean().item()),
                weight_min=float(valid_weights.min().item()),
                weight_max=float(valid_weights.max().item()),
            )
        else:
            self._update_stats(cross_pair_count=cross_pair_count)

        # Update only after current weights have been computed.
        self._update_prototypes(features, labels, domains)
        return pair_weight.detach()

    def _validate_inputs(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        domains: torch.Tensor,
    ) -> None:
        if features.ndim != 2:
            raise ValueError(f"features must be [B, F], got {tuple(features.shape)}")
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {features.shape[1]}"
            )
        if labels.ndim != 1 or domains.ndim != 1:
            raise ValueError("labels and domains must be one-dimensional")
        if not (features.shape[0] == labels.shape[0] == domains.shape[0]):
            raise ValueError("features, labels and domains must share batch size")
        if labels.numel() and (
            labels.min().item() < 0 or labels.max().item() >= self.num_classes
        ):
            raise ValueError("labels contain an out-of-range class index")
        if domains.numel() and (
            domains.min().item() < 0 or domains.max().item() >= self.num_domains
        ):
            raise ValueError("domains contain an out-of-range source-domain slot")

    @torch.no_grad()
    def _update_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        domains: torch.Tensor,
    ) -> None:
        for class_id in labels.unique(sorted=True):
            class_index = int(class_id.item())
            class_mask = labels.eq(class_index)
            for domain_id in domains[class_mask].unique(sorted=True):
                domain_index = int(domain_id.item())
                mask = class_mask & domains.eq(domain_index)
                batch_proto = F.normalize(features[mask].mean(dim=0), dim=0)

                if not bool(self.prototype_valid[class_index, domain_index].item()):
                    self.prototypes[class_index, domain_index].copy_(batch_proto)
                    self.prototype_valid[class_index, domain_index] = True
                else:
                    updated = (
                        self.proto_momentum
                        * self.prototypes[class_index, domain_index]
                        + (1.0 - self.proto_momentum) * batch_proto
                    )
                    self.prototypes[class_index, domain_index].copy_(
                        F.normalize(updated, dim=0)
                    )

    def _update_stats(
        self,
        cross_pair_count: int,
        valid_pair_count: int = 0,
        reliability_mean: float = 1.0,
        weight_mean: float = 1.0,
        weight_min: float = 1.0,
        weight_max: float = 1.0,
    ) -> None:
        self.last_stats = {
            "cross_pairs": float(cross_pair_count),
            "valid_pairs": float(valid_pair_count),
            "valid_pair_fraction": float(valid_pair_count)
            / float(max(cross_pair_count, 1)),
            "reliability_mean": float(reliability_mean),
            "weight_mean": float(weight_mean),
            "weight_min": float(weight_min),
            "weight_max": float(weight_max),
        }


class ReliableCrossDomainSupConLoss(nn.Module):
    """DCCL SupCon with weights only in positive log-probability aggregation.

    The softmax denominator, negative construction, temperature, self-mask,
    optional domain masks and optional added positives are kept identical to
    DCCL's original ``SupConLoss``. ``pair_weight`` is a ``[B, B]`` matrix for
    original samples and is repeated across views internally.
    """

    def __init__(
        self,
        temperature: float = 0.3,
        mask_out: bool = False,
        neg_mix: bool = False,
        not_sup: bool = False,
        contrast_mode: str = "all",
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.neg_mix = neg_mix
        self.base_temperature = temperature
        self.mask_out = mask_out
        self.not_sup = not_sup

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        add_pos: Optional[torch.Tensor] = None,
        pair_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = features.device
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif self.not_sup:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        if self.neg_mix:
            self_dot = (
                torch.sum(contrast_feature * contrast_feature, 1, keepdim=True)
                / self.temperature
            )
            anchor_dot_contrast_neg = 0.5 * anchor_dot_contrast + 0.5 * self_dot

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        mask_out = 1 - mask
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        if self.neg_mix:
            exp_logits = (
                torch.exp(anchor_dot_contrast_neg - logits_max.detach()) * logits_mask
            )
        else:
            exp_logits = torch.exp(logits) * logits_mask
        if neg_mask is not None:
            exp_logits = exp_logits * neg_mask
        if self.mask_out:
            exp_logits = exp_logits * mask_out
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        if pos_mask is not None:
            log_prob = log_prob * pos_mask

        if pair_weight is None:
            expanded_pair_weight = torch.ones_like(mask)
        else:
            if pair_weight.shape != (batch_size, batch_size):
                raise ValueError(
                    "pair_weight must be [B, B], got "
                    f"{tuple(pair_weight.shape)} for B={batch_size}"
                )
            expanded_pair_weight = pair_weight.detach().to(
                device=device, dtype=log_prob.dtype
            ).repeat(anchor_count, contrast_count)

        weighted_positive_mask = mask * expanded_pair_weight

        if add_pos is not None:
            add_logits = (
                torch.sum(add_pos * contrast_feature, 1, keepdim=True)
                / self.temperature
                - logits_max.detach()
            )
            add_logits = torch.squeeze(add_logits)
            mean_log_prob_pos = (
                (weighted_positive_mask * log_prob).sum(1) + add_logits
            ) / (weighted_positive_mask.sum(1) + 1)
        else:
            mean_log_prob_pos = (
                weighted_positive_mask * log_prob
            ).sum(1) / weighted_positive_mask.sum(1)

        loss = -(
            self.temperature / self.base_temperature
        ) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()
