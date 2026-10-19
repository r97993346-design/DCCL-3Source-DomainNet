"""Losses for causal interventional prompt tuning."""

import torch
import torch.nn.functional as F


def classification_loss(logits, labels):
    """Mean label CE over the K intervention-specific predictions."""
    return torch.stack([F.cross_entropy(one_logits, labels) for one_logits in logits.unbind(1)]).mean()


def decomposition_loss_terms(causal_logits, spurious_logits, labels):
    """Return E's label CE and KL(U || p_S) separately, without weights."""
    causal_discrimination = F.cross_entropy(causal_logits, labels)
    uniform_target = torch.full_like(spurious_logits, 1.0 / spurious_logits.shape[-1])
    spurious_uniformity = F.kl_div(
        F.log_softmax(spurious_logits, dim=-1), uniform_target, reduction="batchmean"
    )
    return causal_discrimination, spurious_uniformity


def decomposition_loss(causal_logits, spurious_logits, labels):
    """CIPT discriminative-causal and uniform-spurious decomposition objective."""
    causal_discrimination, spurious_uniformity = decomposition_loss_terms(
        causal_logits, spurious_logits, labels
    )
    return causal_discrimination + spurious_uniformity


def prompt_consistency_loss(logits):
    """Mean KL over all ordered, distinct prompt pairs for each image.

    ``logits`` is [B,K,C], with K >= 2. Both sides remain differentiable;
    neither a detached teacher nor ground-truth labels enter this loss.

    The O(B*K*C) expression below is exactly the ordered-pair average
    sum_{k != l} KL(p_k || p_l) / (B*K*(K-1)), without a [B,K,K,C]
    intermediate. The log-probability mean is NOT an ensemble probability
    target; it is an algebraic reduction of the full pairwise objective.
    """
    if logits.ndim != 3:
        raise ValueError("Prompt logits must have shape [B,K,C].")
    batch, prompts, classes = logits.shape
    if batch < 1 or classes < 1 or prompts < 2:
        raise ValueError("Prompt KL requires B >= 1, K >= 2, and C >= 1.")
    if not logits.is_floating_point():
        raise ValueError("Prompt logits must be floating point.")
    # Avoid underflow in mixed precision, retaining float64 for gradient checks.
    values = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    log_prob = F.log_softmax(values, dim=-1)
    prob = log_prob.exp()
    mean_log_prob = log_prob.mean(dim=1, keepdim=True)
    loss = (prob * (log_prob - mean_log_prob)).sum(dim=-1).mean()
    # Only cancellation at identical predictions can produce a tiny negative.
    return (loss * (prompts / float(prompts - 1))).clamp_min(0.0)


def independence_loss(causal_features, spurious_features, eps=1e-6):
    """Official CIPT independence penalty: 0.5 * mean(cos(e, s)^2)."""
    cosine = F.cosine_similarity(
        causal_features, spurious_features, dim=-1, eps=eps
    )
    return 0.5 * cosine.square().mean()

