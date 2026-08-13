"""Official-aligned losses for Causal Interventional Prompt Tuning (CIPT)."""

import torch
import torch.nn.functional as F


def classification_loss(logits, labels):
    """CIPT Eq. (21): mean CE over K intervention-specific predictions."""
    if logits.ndim != 3:
        raise ValueError("Expected logits with shape [B, K, C], got {}".format(tuple(logits.shape)))
    batch, num_templates, num_classes = logits.shape
    repeated_labels = labels[:, None].expand(batch, num_templates).reshape(-1)
    return F.cross_entropy(logits.reshape(batch * num_templates, num_classes), repeated_labels)


def decomposition_loss(causal_logits, spurious_logits, labels):
    """CIPT Eq. (11): causal CE + KL(uniform || spurious prediction)."""
    if causal_logits.shape != spurious_logits.shape:
        raise ValueError("Causal and spurious logits must have the same shape.")

    causal_discrimination = F.cross_entropy(causal_logits, labels)
    num_classes = causal_logits.shape[-1]
    log_spurious = F.log_softmax(spurious_logits, dim=-1)
    uniform_target = torch.full_like(log_spurious, 1.0 / num_classes)
    # torch.kl_div(log_q, p) computes KL(p || q).
    spurious_uniformity = F.kl_div(log_spurious, uniform_target, reduction="batchmean")
    return causal_discrimination + spurious_uniformity


def independence_loss(causal_features, spurious_features, eps=1e-6):
    """CIPT Eq. (14)-(15): 0.5 * mean(cos(e, s)^2)."""
    cosine = F.cosine_similarity(causal_features, spurious_features, dim=-1, eps=eps)
    return 0.5 * cosine.square().mean()
