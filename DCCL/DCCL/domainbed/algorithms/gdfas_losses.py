"""ICCV 2025 GD-FAS losses adapted from the authors' official models/networks.py.

Source: https://github.com/SeungjinJung/GD-FAS/blob/main/models/networks.py
The input is a multi-class DomainBed batch. Training objectives intentionally
retain the official GS-RM, FOD and image-instance loss reductions.
"""
import math

import torch
import torch.nn.functional as F


def orthogonal_text_basis(text_features, eps=1e-6):
    """Official Gram-Schmidt over ensemble class text embeddings [C, D].

    The original Python implementation keeps a nonzero orthogonal vector per
    class and normalizes it. This version uses the same projection rule.
    """
    vectors = []
    for vector in text_features:
        orthogonal = vector.clone()
        for basis in vectors:
            orthogonal = orthogonal - (torch.dot(vector, basis) / torch.dot(basis, basis).clamp_min(eps)) * basis
        if orthogonal.norm().detach().item() > eps:
            vectors.append(orthogonal / orthogonal.norm())
    if not vectors:
        raise ValueError("GD-FAS requires at least one nondegenerate text vector")
    return torch.stack(vectors, dim=0)


def fod_decomposition(image_features, text_features, detach_projection=True):
    """Projection onto the text basis and its orthogonal residual.

    Official GD-FAS uses stop-gradient on the invariant projection when
    computing the domain-specific residual; preserve this by default.
    """
    basis = orthogonal_text_basis(text_features)
    invariant = (image_features @ basis.t()) @ basis
    specific = image_features - (invariant.detach() if detach_projection else invariant)
    return invariant, specific, basis


def fod_domain_loss(specific, domains, temperature=0.1, official_reduction=True):
    """Domain-supervised InfoNCE on residual S, as in official GD-FAS.

    Important: the released code calls mean() on the entire [N, N-1]
    matrix AFTER dividing each positive by its anchor's positive count.
    This is (N-1) times smaller than usual per-anchor SupCon. Keep it for
    faithful transfer; the optional alternate reduction is for ablations.
    """
    n = specific.shape[0]
    if n <= 1:
        return specific.sum() * 0.0
    valid = ~torch.eye(n, dtype=torch.bool, device=specific.device)
    similarity = (specific @ specific.t()) / temperature
    similarity = similarity.masked_select(valid).view(n, n - 1)
    positives = domains.view(-1, 1).eq(domains.view(1, -1))
    positives = positives.masked_select(valid).view(n, n - 1).to(similarity.dtype)
    counts = positives.sum(dim=1, keepdim=True)
    log_prob = F.log_softmax(similarity, dim=1)
    per_entry = positives * log_prob / counts.clamp_min(1)
    if official_reduction:
        return -per_entry.mean()
    active = counts.squeeze(1) > 0
    return -per_entry.sum(dim=1)[active].mean() if active.any() else specific.sum() * 0.0


def groupwise_scaling_loss(logits, labels, domains, beta=1.5, alpha=None):
    """Official GS-RM with multi-class group id = y + C * domain.

    Preserves per-group CE * group_size/total and the detached NumPy-like
    standardization used by the authors. Missing groups are omitted rather
    than generating invalid cross-entropy on empty groups.
    """
    if alpha is None:
        alpha = math.log(int(domains.max().item()) + 1) / 2
    alpha = max(float(alpha), 1e-6)
    num_classes = logits.shape[1]
    groups = labels.long() + num_classes * domains.long()
    losses = []
    for group in torch.unique(groups):
        indices = groups == group
        n = int(indices.sum().item())
        losses.append(F.cross_entropy(logits[indices], labels[indices]) * n / logits.shape[0])
    losses = torch.stack(losses)
    scores = losses.detach()
    standard = (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-8)
    scaler = beta / (1.0 + torch.exp(-standard / alpha)) - beta / 2.0 + 1.0
    return (scaler * losses).sum()


def paired_image_instance_loss(z, temperature=0.8):
    """Official GD-FAS L_II-SIM: two ordered views per source image.

    z has shape [2B, 256], with view-1 followed by view-2 in the same order.
    The paper's released implementation uses logits / 0.8, not 0.1.
    """
    n = z.shape[0]
    if n % 2 != 0 or n < 4:
        raise ValueError("GD-FAS image-instance loss expects 2B views and B>=2")
    b = n // 2
    z = F.normalize(z, dim=1)
    sims = (z @ z.t()) / temperature
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    sim_flat = sims[mask].view(n, n - 1)
    ids = torch.arange(b, device=z.device).repeat(2)
    pos_flat = ids[:, None].eq(ids[None, :])[mask].view(n, n - 1)
    values = torch.cat(
        (sim_flat[pos_flat].view(n, 1), sim_flat[~pos_flat].view(n, n - 2)), dim=1
    )
    return F.cross_entropy(values, torch.zeros(n, dtype=torch.long, device=z.device))
