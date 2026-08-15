"""Losses for Causal Interventional Prompt Tuning (CIPT)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class CIPTLossOutput:
    """Container returned by :func:`cipt_loss`."""

    loss: Tensor
    classification: Tensor
    decomposition: Tensor
    independence: Tensor
    causal_ce: Tensor
    spurious_kl: Tensor


def tda_classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute Eq. (21), averaging CE over K augmented features.

    Args:
        logits: Interventional logits with shape ``[batch, K, num_classes]``.
        labels: Class ids with shape ``[batch]``.
    """

    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [B, K, C], got {logits.shape}.")
    batch, num_templates, num_classes = logits.shape
    repeated_labels = labels[:, None].expand(batch, num_templates).reshape(-1)
    return F.cross_entropy(logits.reshape(batch * num_templates, num_classes), repeated_labels)


def decomposition_loss(causal_logits: Tensor, spurious_logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Compute Eq. (11): CE on causal features + KL-to-uniform on spurious features.

    The paper writes ``KL(p_uni, p_s) - y log p_e``. In PyTorch,
    ``kl_div(log_softmax(spurious), uniform)`` gives ``KL(p_uni || p_s)``.
    """

    if causal_logits.shape != spurious_logits.shape:
        raise ValueError("Causal and spurious logits must have the same shape.")

    num_classes = causal_logits.shape[-1]
    causal_ce = F.cross_entropy(causal_logits, labels)
    log_spurious = F.log_softmax(spurious_logits, dim=-1)
    uniform = torch.full_like(log_spurious, fill_value=1.0 / num_classes)
    spurious_kl = F.kl_div(log_spurious, uniform, reduction="batchmean")
    return causal_ce + spurious_kl, causal_ce, spurious_kl


def independence_loss(causal_features: Tensor, spurious_features: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute Eq. (14)-(15), a squared cosine-correlation penalty."""

    cov = F.cosine_similarity(causal_features, spurious_features, dim=-1, eps=eps)
    return 0.5 * cov.square().mean()


def cipt_loss(
    interventional_logits: Tensor,
    causal_logits: Tensor,
    spurious_logits: Tensor,
    causal_features: Tensor,
    spurious_features: Tensor,
    labels: Tensor,
    beta: float = 2.0,
    gamma: float = 5.0,
) -> CIPTLossOutput:
    """Combine the CIPT objectives as Eq. (22).

    Args:
        interventional_logits: Logits from TDA, shape ``[B, K, C]``.
        causal_logits: Logits from causal adapter features, shape ``[B, C]``.
        spurious_logits: Logits from spurious adapter features, shape ``[B, C]``.
        causal_features: Causal features ``e``, shape ``[B, D]``.
        spurious_features: Spurious features ``s``, shape ``[B, D]``.
        labels: Class ids, shape ``[B]``.
        beta: Weight for the decomposition loss.
        gamma: Weight for the independence loss.
    """

    classification = tda_classification_loss(interventional_logits, labels)
    decomposition, causal_ce, spurious_kl = decomposition_loss(causal_logits, spurious_logits, labels)
    independence = independence_loss(causal_features, spurious_features)
    total = classification + beta * decomposition + gamma * independence
    return CIPTLossOutput(
        loss=total,
        classification=classification,
        decomposition=decomposition,
        independence=independence,
        causal_ce=causal_ce,
        spurious_kl=spurious_kl,
    )

