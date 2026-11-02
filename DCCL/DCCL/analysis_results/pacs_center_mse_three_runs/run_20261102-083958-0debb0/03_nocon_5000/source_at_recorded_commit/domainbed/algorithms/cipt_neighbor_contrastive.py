"""Detached V/E cross-domain neighborhood diagnostics for anchor weighting.

The neighborhood is used only to set each anchor's SupCon coefficient. It does
not replace the positive mask, negatives, or denominator of the original loss.
"""

import math
from numbers import Integral

import torch
import torch.nn.functional as F


def validate_neighbor_options(k, alpha):
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("cipt_neighbor_k must be a positive integer")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("cipt_neighbor_alpha must be finite and non-negative")


def empty_neighbor_stats(reference):
    """Use nbr_check=0 to distinguish skipped diagnostics from measured zeros."""
    return {
        name: reference.new_zeros(())
        for name in (
            "nbr_check", "nbr_pur_v", "nbr_pur_e", "nbr_gap",
            "nbr_cover", "nbr_drop",
        )
    }


@torch.no_grad()
def neighbor_retention_weights(visual, causal, labels, domains, k=5, alpha=0.5):
    """Return w_i = 1 + alpha * max(r_i(V) - r_i(E), 0).

    Inputs contain only the existing merged SOURCE minibatches. Cosine kNN is
    computed separately in each other source domain and the resulting class
    purities are averaged equally across eligible domains. A gallery domain
    is eligible for an anchor only when it contains both a same-class and a
    different-class sample. With no eligible domain, its weight is exactly 1.

    Both representations use the same gallery and effective k=min(k, size).
    All calculations, including the E neighborhood, are detached. This is a
    minibatch estimate of semantic neighborhood change, not causal inference.
    """
    validate_neighbor_options(k, alpha)
    if visual.ndim != 2 or causal.ndim != 2:
        raise ValueError("visual and causal features must have shape [B, D]")
    batch = causal.shape[0]
    if visual.shape[0] != batch or labels.shape != (batch,) or domains.shape != (batch,):
        raise ValueError("features, labels and source-domain ids must share B")
    if any(t.device != causal.device for t in (visual, labels, domains)):
        raise ValueError("features, labels and source-domain ids must share a device")

    weights = causal.new_ones((batch,), dtype=torch.float32)
    stats = empty_neighbor_stats(weights)
    stats["nbr_check"] = weights.new_ones(())
    if batch < 2:
        return weights, stats

    unique_domains = domains.unique(sorted=True)
    if unique_domains.numel() < 2:
        return weights, stats

    visual = F.normalize(visual.float(), dim=-1)
    causal = F.normalize(causal.float(), dim=-1)
    similarity_v = visual @ visual.t()
    similarity_e = causal @ causal.t()
    purity_v = weights.new_zeros(batch)
    purity_e = weights.new_zeros(batch)
    eligible_count = weights.new_zeros(batch)

    for domain in unique_domains.unbind():
        gallery = torch.nonzero(domains == domain, as_tuple=False).flatten()
        gallery_labels = labels[gallery]
        same_class = labels[:, None].eq(gallery_labels[None, :])
        eligible = (
            domains.ne(domain)
            & same_class.any(dim=1)
            & (~same_class).any(dim=1)
        )
        effective_k = min(k, gallery.numel())
        neighbors_v = similarity_v[:, gallery].topk(effective_k, dim=1).indices
        neighbors_e = similarity_e[:, gallery].topk(effective_k, dim=1).indices
        domain_purity_v = same_class.gather(1, neighbors_v).float().mean(dim=1)
        domain_purity_e = same_class.gather(1, neighbors_e).float().mean(dim=1)
        purity_v += domain_purity_v * eligible
        purity_e += domain_purity_e * eligible
        eligible_count += eligible

    purity_v /= eligible_count.clamp_min(1)
    purity_e /= eligible_count.clamp_min(1)
    valid = eligible_count > 0
    # Clip AFTER averaging domain purities, as specified by r_i(V)-r_i(E).
    gap = (purity_v - purity_e).clamp(min=0, max=1)
    weights += float(alpha) * gap
    count = valid.float().sum().clamp_min(1)
    stats.update({
        "nbr_pur_v": (purity_v * valid).sum() / count,
        "nbr_pur_e": (purity_e * valid).sum() / count,
        "nbr_gap": gap.sum() / count,
        "nbr_cover": valid.float().mean(),
        "nbr_drop": ((gap > 0) & valid).float().sum() / count,
    })
    return weights, stats
