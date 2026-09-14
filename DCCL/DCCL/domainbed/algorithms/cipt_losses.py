"""Losses for causal interventional prompt tuning."""

import math

import torch
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


def intervention_reference_reliability(intervention_logits, labels):
    """Fraction of existing TDA predictions that retain the ground-truth label.

    ``intervention_logits`` is [B, K, C], before averaging over interventions.
    This is a detached training-only proxy, not a causal-identification score.
    Consistently wrong predictions receive zero reliability.
    """
    if intervention_logits.ndim != 3 or intervention_logits.shape[1] < 1:
        raise ValueError("intervention_logits must have shape [B, K>=1, C]")
    labels = labels.reshape(-1)
    if labels.numel() != intervention_logits.shape[0]:
        raise ValueError("intervention_logits and labels must share batch size")
    with torch.no_grad():
        predictions = intervention_logits.detach().argmax(dim=-1)
        return predictions.eq(labels[:, None]).float().mean(dim=1)


def confidence_reference_reliability(causal_logits, labels):
    """Detached correct-class confidence on pre-TDA e, for the control run."""
    if causal_logits.ndim != 2:
        raise ValueError("causal_logits must have shape [B, C]")
    labels = labels.reshape(-1)
    if labels.numel() != causal_logits.shape[0]:
        raise ValueError("causal_logits and labels must share batch size")
    with torch.no_grad():
        probabilities = F.softmax(causal_logits.detach().float(), dim=-1)
        return probabilities.gather(1, labels[:, None]).squeeze(1)


def reference_weighted_causal_contrastive_loss(
    causal_features,
    labels,
    reference_reliability=None,
    temperature=0.1,
    reference_floor=0.1,
):
    """Reference-weighted single-view SupCon, with similarities only on e.

    Let r_j be a detached reference score in [0, 1]. The positive-pair target
    is w_ij = 1[y_i=y_j, i!=j] * (floor + (1-floor)*r_j), normalized over j.
    Only the reference column is weighted: anchors are averaged uniformly.
    The denominator is the original SupCon denominator (all non-self e).
    Both sides of an e/e similarity retain gradients, as in the base branch.

    Uniform scores exactly recover the original loss mathematically. A
    positive floor prevents all-zero scores from dropping hard anchors.
    No prototypes, domain alignment, mining loop, projection or extra view
    is used. Missing positives yield a differentiable zero contribution.
    """
    if causal_features.ndim != 2:
        raise ValueError("causal_features must have shape [B, D]")
    labels = labels.reshape(-1)
    batch_size = causal_features.shape[0]
    if labels.numel() != batch_size:
        raise ValueError("causal_features and labels must share batch size")
    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(reference_floor)) or not 0.0 < reference_floor <= 1.0:
        raise ValueError("reference_floor must be in (0, 1]")

    features = F.normalize(causal_features.float(), dim=-1)
    if reference_reliability is None:
        reliability = features.new_ones(batch_size)
    else:
        if reference_reliability.numel() != batch_size:
            raise ValueError("reference_reliability must have one score per sample")
        reliability = reference_reliability.detach().reshape(-1).to(features).clamp(0, 1)

    zero = features.sum() * 0.0
    metrics = {
        "valid_anchor_fraction": features.new_zeros(()),
        "irc_reliability_mean": reliability.mean() if batch_size else features.new_zeros(()),
        "irc_reliability_std": reliability.std(unbiased=False) if batch_size else features.new_zeros(()),
        "irc_positive_ess_fraction": features.new_zeros(()),
        "irc_weighted_anchor_fraction": features.new_zeros(()),
    }
    if batch_size <= 1:
        return zero, metrics

    self_mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~self_mask
    positive_count = positives.sum(dim=1)
    valid = positive_count.gt(0).float()
    valid_count = valid.sum().clamp_min(1.0)

    # One dense similarity matrix; no data-dependent GPU-to-Python branches.
    similarities = features @ features.t() / temperature
    denominator = similarities.masked_fill(self_mask, torch.finfo(features.dtype).min)
    log_probabilities = similarities - torch.logsumexp(denominator, dim=1, keepdim=True)

    reference_weights = reference_floor + (1.0 - reference_floor) * reliability
    weights = positives.float() * reference_weights[None, :]
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    anchor_losses = -(weights * log_probabilities).sum(dim=1)
    loss = (anchor_losses * valid).sum() / valid_count

    with torch.no_grad():
        # ESS / positive count = 1 for uniform references. This diagnoses
        # whether varying scores actually change any anchor's positive weights.
        ess_fraction = (
            weights.square().sum(dim=1) * positive_count.float()
        ).clamp_min(1e-12).reciprocal()
        uniform = positives.float() / positive_count.clamp_min(1).float()[:, None]
        weighted = (weights - uniform).abs().amax(dim=1).gt(1e-6).float()
        metrics.update({
            "valid_anchor_fraction": valid.mean(),
            "irc_positive_ess_fraction": (ess_fraction * valid).sum() / valid_count,
            "irc_weighted_anchor_fraction": (weighted * valid).sum() / valid_count,
        })
    return loss, metrics
