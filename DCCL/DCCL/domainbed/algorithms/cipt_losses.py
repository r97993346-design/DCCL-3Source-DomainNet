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
