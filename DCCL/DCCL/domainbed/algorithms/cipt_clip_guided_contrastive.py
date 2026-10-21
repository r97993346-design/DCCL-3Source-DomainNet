"""Single-view, projection-free cross-source-domain SupCon on causal E.

Frozen visual V supplies only positive selection and detached pair weights.
Same-domain same-class pairs and unselected same-class pairs are neutral.
No text, target-domain data, second view or extra encoder call is used here.
"""

import math
from numbers import Integral

import torch
import torch.nn.functional as F


def validate_clip_teacher_options(temperature, mix, topk):
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("cipt_clip_teacher_temperature must be finite and positive")
    if not math.isfinite(mix) or not 0.0 <= mix <= 1.0:
        raise ValueError("cipt_clip_teacher_mix must be finite and in [0, 1]")
    if isinstance(topk, bool) or not isinstance(topk, Integral) or topk < 0:
        raise ValueError("cipt_clip_teacher_topk must be a non-negative integer")


def empty_clip_guided_stats(reference):
    return {
        name: reference.detach().new_zeros((), dtype=torch.float32)
        for name in (
            "cg_contrastive_loss", "cg_valid_anchor_fraction",
            "cg_cross_domain_pos_count", "cg_teacher_pos_cos_mean",
            "cg_teacher_weight_entropy", "cg_teacher_weight_max", "cg_active",
        )
    }


def source_domain_ids(labels, label_batches):
    """Match torch.cat(y)'s order; y contains one minibatch per SOURCE only."""
    if not label_batches:
        raise ValueError("label_batches must contain SOURCE minibatches")
    if any(y.ndim != 1 or y.device != labels.device for y in label_batches):
        raise ValueError("SOURCE labels must be 1D tensors on labels.device")
    if not torch.equal(torch.cat(label_batches), labels):
        raise ValueError("labels must equal the ordered concatenation of label_batches")
    return torch.cat([
        torch.full_like(y, domain, dtype=torch.long)
        for domain, y in enumerate(label_batches)
    ])


@torch.no_grad()
def cross_domain_targets(visual, labels, domain_ids, teacher_temperature=0.2,
                         teacher_mix=0.5, topk=0):
    """Return detached (weights, positives, denominator_mask, teacher_cosines).

    Positive candidates: same class, different source domain, not self.
    Top-K is per anchor, across all eligible source domains, using V only.
    Negatives: all different-class samples, including the anchor's own domain.
    Only selected positives and negatives enter the denominator.
    """
    validate_clip_teacher_options(teacher_temperature, teacher_mix, topk)
    if visual.ndim != 2 or visual.shape[1] < 1 or not visual.is_floating_point():
        raise ValueError("visual must be floating point with shape [B,D], D >= 1")
    batch = visual.shape[0]
    if labels.shape != (batch,) or domain_ids.shape != (batch,):
        raise ValueError("labels and domain_ids must have shape [B]")
    if labels.dtype != torch.long or domain_ids.dtype != torch.long:
        raise ValueError("labels and domain_ids must be LongTensors")
    if labels.device != visual.device or domain_ids.device != visual.device:
        raise ValueError("visual, labels and domain_ids must share a device")
    if not torch.isfinite(visual).all():
        raise ValueError("visual must contain finite features")

    # Explicitly disable autocast: .float() alone does not protect matmul.
    with torch.autocast(device_type=visual.device.type, enabled=False):
        unit_v = F.normalize(visual.detach().float(), dim=-1)
        similarity = unit_v @ unit_v.t()
        same_class = labels[:, None].eq(labels[None, :])
        different_domain = domain_ids[:, None].ne(domain_ids[None, :])
        not_self = ~torch.eye(batch, dtype=torch.bool, device=visual.device)
        positives = same_class & different_domain & not_self

        if topk > 0 and batch:
            indices = similarity.masked_fill(~positives, -torch.inf).topk(
                min(topk, batch), dim=1
            ).indices
            selected = torch.zeros_like(positives).scatter_(1, indices, True)
            positives = positives & selected

        denominator = positives | (~same_class & not_self)
        weights = torch.zeros_like(similarity)
        counts = positives.sum(dim=1)
        valid = counts > 0
        # Never softmax an all-masked row (which would produce NaNs).
        if valid.any():
            mask = positives[valid]
            uniform = mask.float() / counts[valid, None].float()
            if teacher_mix == 0.0:
                weights[valid] = uniform
            else:
                scores = similarity[valid].masked_fill(~mask, -torch.inf)
                scores = scores - scores.max(dim=1, keepdim=True).values
                teacher = F.softmax(scores / teacher_temperature, dim=1)
                weights[valid] = (1.0 - teacher_mix) * uniform + teacher_mix * teacher
    return weights, positives, denominator, similarity


def clip_guided_cross_domain_loss(visual, causal, labels, domain_ids,
                                  temperature=0.1, teacher_temperature=0.2,
                                  teacher_mix=0.5, topk=0):
    """Return (loss, detached metrics); mix=0 is equal-weight cross-domain SupCon.

    L_i = -sum_j w_ij log(exp(cos(E_i,E_j)/tau) /
                             sum_{a in P_i union N_i} exp(cos(E_i,E_a)/tau)).
    Average over anchors with at least one selected positive. Weights sum to
    one per valid anchor; positive count is never an extra loss coefficient.
    """
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("cipt_contrastive_temperature must be finite and positive")
    if causal.ndim != 2 or causal.shape[1] < 1 or not causal.is_floating_point():
        raise ValueError("causal must be floating point with shape [B,D], D >= 1")
    if visual.shape[0] != causal.shape[0] or visual.device != causal.device:
        raise ValueError("visual and causal must share batch size and device")
    if not torch.isfinite(causal).all():
        raise ValueError("causal must contain finite features")

    weights, positives, denominator, similarity = cross_domain_targets(
        visual, labels, domain_ids, teacher_temperature, teacher_mix, topk
    )
    stats = empty_clip_guided_stats(causal)
    counts = positives.sum(dim=1)
    valid = counts > 0
    with torch.autocast(device_type=causal.device.type, enabled=False):
        features = F.normalize(causal.float(), dim=-1)
        if not valid.any():
            # Includes B=0/1, one source domain and no shared cross-domain class.
            # Keep a graph to E so zero.backward() is safe.
            return features.sum() * 0.0, stats

        # Only valid rows enter logsumexp, preventing all--inf row gradients.
        logits = features[valid] @ features.t() / temperature
        log_denominator = torch.logsumexp(
            logits.masked_fill(~denominator[valid], -torch.inf), dim=1, keepdim=True
        )
        # Subtract from unmasked logits: neutral entries remain finite before
        # multiplication by zero, avoiding 0 * -inf.
        loss = -(weights[valid] * (logits - log_denominator)).sum(dim=1).mean()

    with torch.no_grad():
        valid_weights = weights[valid]
        entropy = -(valid_weights * valid_weights.clamp_min(1e-30).log()).sum(dim=1)
        stats.update({
            "cg_contrastive_loss": loss.detach(),
            "cg_valid_anchor_fraction": valid.float().mean(),
            "cg_cross_domain_pos_count": counts[valid].float().mean(),
            "cg_teacher_pos_cos_mean": similarity[positives].mean(),
            "cg_teacher_weight_entropy": entropy.mean(),
            "cg_teacher_weight_max": valid_weights.max(),
            "cg_active": features.new_ones(()),
        })
    return loss, stats
