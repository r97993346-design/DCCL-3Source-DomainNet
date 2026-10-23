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


def relative_distance_loss(causal_features, class_anchors, text_bank, labels, indices,
                           anchor_mode="full"):
    """Mean squared error between student/teacher cosine relationships.

    E: [B,D], frozen A: [C,D], frozen T: [C,80,D], y: [B], J: [B,n].
    Only E is differentiable. J was selected without labels; labels are used
    here solely to look up the correct class's anchor and selected templates.
    ``full`` keeps the original all-80 class center. ``selected`` instead
    averages the same n frozen vectors used by this image's RD relationships,
    then normalizes that per-image center. Neither teacher receives gradients.
    This preserves all n relationships rather than minimizing MSE(E, A[y]).
    """
    if anchor_mode not in ("full", "selected"):
        raise ValueError("anchor_mode must be full or selected")
    if causal_features.ndim != 2 or causal_features.shape[0] < 1:
        raise ValueError("causal_features must have nonempty shape [B,D]")
    batch, dim = causal_features.shape
    if (text_bank.ndim != 3 or text_bank.shape[0] < 1
            or text_bank.shape[1:] != (80, dim)):
        raise ValueError("text_bank must have shape [C,80,D]")
    if class_anchors.shape != (text_bank.shape[0], dim):
        raise ValueError("class_anchors must have shape [C,D]")
    if labels.shape != (batch,) or labels.dtype != torch.long:
        raise ValueError("labels must be a LongTensor with shape [B]")
    if (indices.ndim != 2 or indices.shape[0] != batch
            or not 2 <= indices.shape[1] <= 80 or indices.dtype != torch.long):
        raise ValueError("indices must be a LongTensor with shape [B,n], 2 <= n <= 80")
    if (labels < 0).any() or (labels >= text_bank.shape[0]).any():
        raise ValueError("labels are outside the class bank")
    if (indices < 0).any() or (indices >= 80).any():
        raise ValueError("template indices must be between 0 and 79")
    unit_e = F.normalize(causal_features.float(), dim=-1)
    selected_text = text_bank.detach().float()[labels[:, None], indices]
    unit_t = F.normalize(selected_text, dim=-1)
    if anchor_mode == "selected":
        unit_a = F.normalize(selected_text.mean(dim=1), dim=-1)
    else:
        unit_a = F.normalize(class_anchors.detach().float()[labels], dim=-1)
    student_similarity = (unit_e[:, None, :] * unit_t).sum(dim=-1)
    teacher_similarity = (unit_a[:, None, :] * unit_t).sum(dim=-1)
    return (student_similarity - teacher_similarity).square().mean()


def independence_loss(causal_features, spurious_features, eps=1e-6):
    """Official CIPT independence penalty: 0.5 * mean(cos(e, s)^2)."""
    cosine = F.cosine_similarity(
        causal_features, spurious_features, dim=-1, eps=eps
    )
    return 0.5 * cosine.square().mean()


