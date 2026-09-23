"""Losses for causal interventional prompt tuning."""

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


def cross_correlation_loss(causal_features, spurious_features, eps=1e-4):
    """Cross-dimensional decorrelation between causal E and spurious S.

    Each feature dimension is standardized over the current batch, then the
    D x D cross-correlation matrix C_ES is driven toward zero. This is stronger
    than the original sample-wise cosine orthogonality while remaining a
    lightweight linear-dependence penalty.
    """
    if causal_features.ndim != 2 or spurious_features.ndim != 2:
        raise ValueError("Expected causal/spurious features with shape [B, D].")
    if causal_features.shape != spurious_features.shape:
        raise ValueError("Causal and spurious features must have the same shape.")

    causal = causal_features.float()
    spurious = spurious_features.float()
    causal_centered = causal - causal.mean(dim=0, keepdim=True)
    spurious_centered = spurious - spurious.mean(dim=0, keepdim=True)

    causal_scale = causal_centered.pow(2).mean(dim=0, keepdim=True).add(eps).sqrt()
    spurious_scale = spurious_centered.pow(2).mean(dim=0, keepdim=True).add(eps).sqrt()
    causal_norm = causal_centered / causal_scale
    spurious_norm = spurious_centered / spurious_scale

    batch_size = max(int(causal_norm.shape[0]), 1)
    cross_corr = causal_norm.t().matmul(spurious_norm) / float(batch_size)
    return cross_corr.square().mean()
