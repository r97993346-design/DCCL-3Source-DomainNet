"""Small, auditable causal components used by PICCL.

The components in this module never change DCCL's classifier or contrastive
definitions.  They only learn a low-rank intervention-sensitive subspace and
remove its projection from pooled backbone features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value from {value!r}")
    return bool(value)


class PairedInterventionResponseEstimator(nn.Module):
    """Excess augmentation response relative to a frozen reference encoder."""

    def forward(self, z, z_int, z_ref, z_int_ref):
        tensors = (z, z_int, z_ref, z_int_ref)
        if any(tensor.dim() != 2 for tensor in tensors):
            raise ValueError("PIRE inputs must be pooled feature tensors [B,D]")
        if len({tuple(tensor.shape) for tensor in tensors}) != 1:
            raise ValueError("PIRE inputs must have identical shapes")
        return (z_int - z) - (z_int_ref - z_ref).detach()


class ClassDomainResidualBank(nn.Module):
    """EMA class/domain residual prototypes with minimum-support filtering."""

    def __init__(
        self,
        num_classes,
        num_domains,
        feature_dim,
        momentum=0.99,
        min_count=8,
        min_valid_domains=2,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_domains = int(num_domains)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.min_count = int(min_count)
        self.min_valid_domains = int(min_valid_domains)
        self.register_buffer(
            "prototypes",
            torch.zeros(self.num_classes, self.num_domains, self.feature_dim),
        )
        self.register_buffer(
            "initialized",
            torch.zeros(self.num_classes, self.num_domains, dtype=torch.bool),
        )
        self.register_buffer(
            "counts",
            torch.zeros(self.num_classes, self.num_domains, dtype=torch.long),
        )
        self.register_buffer(
            "update_counts",
            torch.zeros(self.num_classes, self.num_domains, dtype=torch.long),
        )

    @torch.no_grad()
    def update(self, residual, labels, domains, min_norm=0.0):
        residual = residual.detach()
        labels = labels.detach().long()
        domains = domains.detach().long()
        if residual.dim() != 2 or residual.shape[1] != self.feature_dim:
            raise ValueError("residual must have shape [B,D]")
        if labels.shape[0] != residual.shape[0] or domains.shape[0] != residual.shape[0]:
            raise ValueError("labels/domains must match residual batch size")

        valid_norm = residual.norm(dim=1) >= float(min_norm)
        for class_id in labels.unique().tolist():
            if not 0 <= class_id < self.num_classes:
                continue
            for domain_id in domains.unique().tolist():
                if not 0 <= domain_id < self.num_domains:
                    continue
                mask = (labels == class_id) & (domains == domain_id) & valid_norm
                if not mask.any():
                    continue
                current = residual[mask].mean(dim=0)
                if self.initialized[class_id, domain_id]:
                    self.prototypes[class_id, domain_id].mul_(self.momentum).add_(
                        current, alpha=1.0 - self.momentum
                    )
                else:
                    self.prototypes[class_id, domain_id].copy_(current)
                    self.initialized[class_id, domain_id] = True
                self.counts[class_id, domain_id].add_(int(mask.sum().item()))
                self.update_counts[class_id, domain_id].add_(1)

    def domain_responses(self):
        responses = []
        for class_id in range(self.num_classes):
            valid = self.initialized[class_id] & (
                self.counts[class_id] >= self.min_count
            )
            if int(valid.sum().item()) < self.min_valid_domains:
                continue
            prototypes = self.prototypes[class_id, valid]
            responses.append(prototypes - prototypes.mean(dim=0, keepdim=True))
        if not responses:
            return self.prototypes.new_zeros((0, self.feature_dim))
        return torch.cat(responses, dim=0).detach()


class InterventionSensitiveSubspace(nn.Module):
    """Learnable raw basis whose QR factor is used for every projection."""

    def __init__(self, feature_dim, rank=16, eps=1e-8):
        super().__init__()
        feature_dim = int(feature_dim)
        rank = min(int(rank), feature_dim)
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.basis = nn.Parameter(torch.randn(feature_dim, rank) * 0.02)
        self.eps = float(eps)

    def orthonormal_basis(self, detach=False, dtype=None):
        q = torch.linalg.qr(self.basis.float(), mode="reduced").Q
        if detach:
            q = q.detach()
        if dtype is not None:
            q = q.to(dtype=dtype)
        return q

    def project(self, values, detach_basis=True):
        if values.numel() == 0:
            return values
        q = self.orthonormal_basis(detach=detach_basis, dtype=values.dtype).to(
            values.device
        )
        return (values @ q) @ q.transpose(0, 1)

    def coverage_loss(self, responses, min_norm=0.0):
        """Mean unexplained directional energy of intervention responses."""
        if responses.numel() == 0:
            return self.basis.sum() * 0.0
        responses = responses.detach()
        norms = responses.norm(dim=1)
        valid = norms > max(float(min_norm), self.eps)
        if not valid.any():
            return self.basis.sum() * 0.0
        directions = responses[valid] / norms[valid].unsqueeze(1).clamp_min(self.eps)
        residual = directions - self.project(directions, detach_basis=False)
        return residual.pow(2).sum(dim=1).mean()

    def orthogonality_loss(self):
        """Condition the raw basis; the forward projection still uses exact QR."""
        normalized = F.normalize(self.basis.float(), dim=0, eps=self.eps)
        gram = normalized.transpose(0, 1) @ normalized
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return (gram - identity).pow(2).mean().to(self.basis.dtype)

    @torch.no_grad()
    def diagnostics(self):
        q = self.orthonormal_basis(detach=True)
        normalized = F.normalize(self.basis.float(), dim=0, eps=self.eps)
        gram = normalized.transpose(0, 1) @ normalized
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return {
            "basis_orthogonality_error": (gram - identity).pow(2).mean(),
            "basis_rank": int(torch.linalg.matrix_rank(q).item()),
            "basis_norm": self.basis.norm(),
        }


class CausalMediatorProjection(nn.Module):
    """Low-disturbance projection; beta=0 returns the exact input object."""

    def forward(self, z, subspace, beta):
        beta = torch.as_tensor(beta, device=z.device, dtype=z.dtype)
        if beta.numel() != 1:
            raise ValueError("beta must be scalar")
        if float(beta.detach().item()) == 0.0:
            return z
        sensitive = subspace.project(z, detach_basis=True)
        return z - beta * sensitive


@torch.no_grad()
def causal_pair_reliability(
    z,
    labels,
    domains,
    subspace,
    min_delta_norm=1e-6,
):
    """Sensitive-energy ratio for cross-domain, same-class sample pairs."""
    labels = labels.detach().long()
    domains = domains.detach().long()
    same_class = labels[:, None].eq(labels[None, :])
    cross_domain = domains[:, None].ne(domains[None, :])
    cross_positive = same_class & cross_domain
    reliability = torch.ones((z.shape[0], z.shape[0]), device=z.device, dtype=z.dtype)
    pair_i, pair_j = cross_positive.nonzero(as_tuple=True)
    if pair_i.numel() == 0:
        empty = z.new_empty(0)
        return reliability, cross_positive, empty, empty

    delta = z.detach()[pair_i] - z.detach()[pair_j]
    sensitive = subspace.project(delta, detach_basis=True)
    delta_energy = delta.square().sum(dim=1)
    raw = sensitive.square().sum(dim=1) / delta_energy.clamp_min(
        torch.finfo(z.dtype).eps
    )
    delta_norm = delta_energy.sqrt()
    raw = torch.where(
        delta_norm < float(min_delta_norm), torch.ones_like(raw), raw
    ).clamp(0.0, 1.0)
    reliability[pair_i, pair_j] = raw
    reliability = 0.5 * (reliability + reliability.transpose(0, 1))
    return reliability.detach(), cross_positive, raw.detach(), delta_norm.detach()


@torch.no_grad()
def reliable_positive_weights(
    pair_reliability,
    labels,
    domains,
    gamma,
    min_weight=0.5,
    num_views=2,
):
    """Expand sample-pair reliability in SupCon's view-major ordering."""
    batch_size = labels.shape[0]
    sample_ids = torch.arange(batch_size, device=labels.device).repeat(num_views)
    expanded_labels = labels.repeat(num_views)
    expanded_domains = domains.repeat(num_views)
    expanded_views = torch.arange(num_views, device=labels.device).repeat_interleave(
        batch_size
    )

    same_class = expanded_labels[:, None].eq(expanded_labels[None, :])
    cross_domain_positive = same_class & expanded_domains[:, None].ne(
        expanded_domains[None, :]
    )
    self_augmentation = sample_ids[:, None].eq(sample_ids[None, :]) & expanded_views[
        :, None
    ].ne(expanded_views[None, :])

    expanded_reliability = pair_reliability[
        sample_ids[:, None], sample_ids[None, :]
    ]
    clipped = expanded_reliability.clamp(float(min_weight), 1.0)
    effective = 1.0 + float(gamma) * (clipped - 1.0)
    weights = torch.ones_like(expanded_reliability)
    weights = torch.where(cross_domain_positive, effective, weights)
    weights = torch.where(self_augmentation, torch.ones_like(weights), weights)
    return (
        weights.detach(),
        cross_domain_positive,
        self_augmentation,
        expanded_reliability.detach(),
    )
