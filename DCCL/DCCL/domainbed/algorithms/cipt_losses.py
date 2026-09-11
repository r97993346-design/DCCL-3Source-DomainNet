"""Losses for causal interventional prompt tuning."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def classification_loss(logits, labels):
    """Mean CE over the K intervention-specific predictions (the sole task CE)."""
    return torch.stack([F.cross_entropy(one_logits, labels) for one_logits in logits.unbind(1)]).mean()


def decomposition_loss(causal_logits, spurious_logits, labels):
    """CIPT discriminative-causal and uniform-spurious decomposition objective."""
    causal_discrimination = F.cross_entropy(causal_logits, labels)
    uniform_target = torch.full_like(spurious_logits, 1.0 / spurious_logits.shape[-1])
    spurious_uniformity = F.kl_div(
        F.log_softmax(spurious_logits, dim=-1), uniform_target, reduction="batchmean"
    )
    return causal_discrimination + spurious_uniformity


def independence_loss(causal_features, spurious_features, eps=1e-6):
    """Official CIPT independence penalty: 0.5 * mean(cos(e, s)^2)."""
    cosine = F.cosine_similarity(
        causal_features, spurious_features, dim=-1, eps=eps
    )
    return 0.5 * cosine.square().mean()


def weakest_domain_bridge_contrastive_loss(
    causal_features,
    labels,
    domain_ids,
    topk=2,
    temperature=0.1,
    margin=0.1,
    eps=1e-8,
):
    """Contrast source domains through bridge pairs in causal space only.

    For every anchor ``e_i``, the loss first finds the Top-K most similar
    same-class causal representations in each *other* source domain.  Their
    mean is that domain's bridge score.  A smooth minimum emphasizes the
    weakest available source-domain bridge, while a smooth maximum emphasizes
    confusing different-class causal representations.  The ranking objective
    requires the weakest positive bridge to exceed the hard-negative score by
    ``margin``.

    Empty class/domain groups are skipped.  An anchor contributes only when it
    has at least one cross-domain same-class bridge and one different-class
    negative.  No augmented view, spurious representation, prompt feature, or
    post-intervention feature is consumed by this loss.

    Returns:
        ``(loss, metrics)`` where every metric is a scalar tensor.
    """
    if causal_features.ndim != 2:
        raise ValueError(
            "causal_features must have shape [B,D], got {}".format(
                tuple(causal_features.shape)
            )
        )

    labels = labels.reshape(-1)
    domain_ids = domain_ids.reshape(-1)
    batch_size = causal_features.shape[0]
    if labels.numel() != batch_size or domain_ids.numel() != batch_size:
        raise ValueError(
            "causal_features, labels, and domain_ids must share batch size"
        )
    if int(topk) < 1:
        raise ValueError("topk must be at least 1")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if float(margin) < 0.0:
        raise ValueError("margin must be non-negative")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")

    topk = int(topk)
    temperature = float(temperature)
    margin = float(margin)
    features = F.normalize(causal_features.float(), dim=-1, eps=eps)
    similarity = features @ features.t()
    zero = features.sum() * 0.0

    anchor_losses = []
    weakest_positive_scores = []
    hard_negative_scores = []
    violations = []
    domain_coverages = []
    unique_domains = torch.unique(domain_ids)

    for anchor_index in range(batch_size):
        anchor_domain = domain_ids[anchor_index]
        other_domains = unique_domains[unique_domains != anchor_domain]
        bridge_scores = []

        for other_domain in other_domains:
            positive_mask = (
                labels.eq(labels[anchor_index])
                & domain_ids.eq(other_domain)
            )
            positive_similarity = similarity[anchor_index][positive_mask]
            if positive_similarity.numel() == 0:
                continue

            selected_count = min(topk, positive_similarity.numel())
            bridge_scores.append(
                positive_similarity.topk(selected_count, largest=True).values.mean()
            )

        possible_domain_count = max(int(other_domains.numel()), 1)
        domain_coverages.append(
            features.new_tensor(len(bridge_scores) / possible_domain_count)
        )

        negative_mask = labels.ne(labels[anchor_index])
        negative_similarity = similarity[anchor_index][negative_mask]
        if not bridge_scores or negative_similarity.numel() == 0:
            continue

        bridge_scores = torch.stack(bridge_scores)
        weakest_positive = -temperature * (
            torch.logsumexp(-bridge_scores / temperature, dim=0)
            - math.log(bridge_scores.numel())
        )
        hard_negative = temperature * (
            torch.logsumexp(negative_similarity / temperature, dim=0)
            - math.log(negative_similarity.numel())
        )
        ranking_gap = hard_negative - weakest_positive + margin

        anchor_losses.append(F.softplus(ranking_gap / temperature))
        weakest_positive_scores.append(weakest_positive)
        hard_negative_scores.append(hard_negative)
        violations.append(ranking_gap.detach().gt(0).float())

    valid_anchor_fraction = features.new_tensor(
        len(anchor_losses) / max(batch_size, 1)
    )
    domain_coverage = (
        torch.stack(domain_coverages).mean()
        if domain_coverages
        else features.new_zeros(())
    )

    if not anchor_losses:
        return zero, {
            "valid_anchor_fraction": valid_anchor_fraction,
            "domain_coverage_fraction": domain_coverage,
            "weakest_positive_similarity": features.new_zeros(()),
            "hard_negative_similarity": features.new_zeros(()),
            "violation_fraction": features.new_zeros(()),
        }

    loss = torch.stack(anchor_losses).mean()
    metrics = {
        "valid_anchor_fraction": valid_anchor_fraction,
        "domain_coverage_fraction": domain_coverage,
        "weakest_positive_similarity": torch.stack(
            weakest_positive_scores
        ).mean().detach(),
        "hard_negative_similarity": torch.stack(
            hard_negative_scores
        ).mean().detach(),
        "violation_fraction": torch.stack(violations).mean(),
    }
    return loss, metrics


class WeakestDomainBridgeContrastiveLoss(nn.Module):
    """Reusable WBC-CL module whose only feature input is causal ``e``."""

    def __init__(
        self,
        topk=2,
        temperature=0.1,
        margin=0.1,
        eps=1e-8,
    ):
        super().__init__()
        if int(topk) < 1:
            raise ValueError("topk must be at least 1")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        if float(margin) < 0.0:
            raise ValueError("margin must be non-negative")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.topk = int(topk)
        self.temperature = float(temperature)
        self.margin = float(margin)
        self.eps = float(eps)

    def forward(self, causal_features, labels, domain_ids):
        return weakest_domain_bridge_contrastive_loss(
            causal_features,
            labels,
            domain_ids,
            topk=self.topk,
            temperature=self.temperature,
            margin=self.margin,
            eps=self.eps,
        )

    def extra_repr(self):
        return "topk={}, temperature={}, margin={}".format(
            self.topk,
            self.temperature,
            self.margin,
        )
