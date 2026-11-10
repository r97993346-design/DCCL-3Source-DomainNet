"""Detached intervention observations and a shared low-rank projection for DICM.

Only ISR differentiates through QR. Task losses treat Q as a fixed projection;
neither observations nor EMA buffers retain an encoder computation graph.
"""

import math

import torch
from torch import nn


class PairedInterventionResponseEstimator(nn.Module):
    def forward(self, z, z_aug, z_ref, z_aug_ref):
        values = (z, z_aug, z_ref, z_aug_ref)
        if any(v.ndim != 2 for v in values) or len({v.shape for v in values}) != 1:
            raise ValueError("PIRE requires four matching pooled [N,D] tensors")
        return ((z_aug - z) - (z_aug_ref - z_ref).detach()).detach()


class ClassDomainResidualBank(nn.Module):
    def __init__(self, num_classes, num_domains, feature_dim, momentum=0.95,
                 min_count=8, min_updates=3, min_valid_domains=None):
        super().__init__()
        if not 0 <= momentum < 1:
            raise ValueError("EMA momentum must be in [0,1)")
        if min_count < 1 or min_updates < 1:
            raise ValueError("Prototype support thresholds must be positive")
        self.num_classes = int(num_classes)
        self.num_domains = int(num_domains)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.min_count = int(min_count)
        self.min_updates = int(min_updates)
        self.min_valid_domains = int(min_valid_domains or num_domains)
        if not 2 <= self.min_valid_domains <= self.num_domains:
            raise ValueError("CDRM requires at least two valid source domains")
        shape = (self.num_classes, self.num_domains)
        self.register_buffer("prototypes", torch.zeros(*shape, self.feature_dim))
        self.register_buffer("initialized", torch.zeros(shape, dtype=torch.bool))
        self.register_buffer("counts", torch.zeros(shape, dtype=torch.long))
        self.register_buffer("update_counts", torch.zeros(shape, dtype=torch.long))
        self.register_buffer("last_update_step", torch.full(shape, -1, dtype=torch.long))

    @torch.no_grad()
    def update(self, residual, labels, domains, step, min_norm=1e-4):
        if residual.ndim != 2 or residual.shape[1] != self.feature_dim:
            raise ValueError("Residuals must have shape [N,D]")
        labels, domains = labels.detach().long(), domains.detach().long()
        if labels.shape != residual.shape[:1] or domains.shape != labels.shape:
            raise ValueError("Labels/domains must match the residual batch")
        if ((labels < 0) | (labels >= self.num_classes)).any():
            raise ValueError("Invalid class ID in source residuals")
        if ((domains < 0) | (domains >= self.num_domains)).any():
            raise ValueError("Use local source IDs 0..num_domains-1, not held-out/global IDs")
        residual = residual.detach().float()
        if not torch.isfinite(residual).all():
            raise FloatingPointError("Non-finite source residual")
        valid = residual.norm(dim=1) > float(min_norm)
        for c in labels.unique().tolist():
            for d in domains.unique().tolist():
                mask = (labels == c) & (domains == d) & valid
                count = int(mask.sum().item())
                if not count:
                    continue
                current = residual[mask].mean(0).to(self.prototypes.dtype)
                if self.initialized[c, d]:
                    self.prototypes[c, d].lerp_(current, 1.0 - self.momentum)
                else:
                    self.prototypes[c, d].copy_(current)
                    self.initialized[c, d] = True
                self.counts[c, d] += count
                self.update_counts[c, d] += 1
                self.last_update_step[c, d] = int(step)

    def valid_mask(self):
        return (self.initialized & (self.counts >= self.min_count)
                & (self.update_counts >= self.min_updates))

    @torch.no_grad()
    def domain_responses(self):
        responses = []
        for c, valid in enumerate(self.valid_mask()):
            if int(valid.sum().item()) < self.min_valid_domains:
                continue
            prototypes = self.prototypes[c, valid].float()
            responses.append(prototypes - prototypes.mean(0, keepdim=True))
        if not responses:
            return self.prototypes.new_empty((0, self.feature_dim), dtype=torch.float32)
        return torch.cat(responses).detach()


class InterventionSensitiveSubspace(nn.Module):
    def __init__(self, feature_dim, rank=8, seed=0):
        super().__init__()
        if not 1 <= int(rank) < int(feature_dim):
            raise ValueError("DICM rank must satisfy 1 <= rank < feature_dim")
        # A private CPU generator preserves shared DCCL head/sampler RNG state.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        raw = torch.randn(int(feature_dim), int(rank), generator=generator)
        self.basis = nn.Parameter(torch.linalg.qr(raw, mode="reduced").Q.contiguous())

    def orthonormal_basis(self, detach=False):
        with torch.autocast(device_type=self.basis.device.type, enabled=False):
            q = torch.linalg.qr(self.basis.float(), mode="reduced").Q
        return q.detach() if detach else q

    def project(self, values, q=None, detach_basis=True):
        q = self.orthonormal_basis() if q is None else q
        if detach_basis:
            q = q.detach()
        with torch.autocast(device_type=values.device.type, enabled=False):
            projected = (values.float() @ q.float()) @ q.float().T
        return projected.to(values.dtype)

    def coverage_loss(self, responses, q=None, min_norm=1e-4):
        """Return mean directional residual energy and its valid sample count."""
        responses = responses.detach().float()
        if responses.ndim != 2 or responses.shape[1] != self.basis.shape[0]:
            raise ValueError("ISR observations must have shape [N,D]")
        if not torch.isfinite(responses).all():
            raise FloatingPointError("Non-finite ISR observation")
        norms = responses.norm(dim=1)
        valid = norms > max(float(min_norm), 1e-12)
        count = int(valid.sum().item())
        if not count:
            # No synthetic zero gradient: Adam must not move B on old momentum.
            return self.basis.new_zeros(()), 0
        directions = responses[valid] / norms[valid, None]
        unexplained = directions - self.project(directions, q=q, detach_basis=False)
        return unexplained.square().sum(1).mean(), count

    @torch.no_grad()
    def diagnostics(self, q=None):
        q = self.orthonormal_basis(detach=True) if q is None else q.detach()
        singular = torch.linalg.svdvals(self.basis.detach().float())
        eye = torch.eye(q.shape[1], device=q.device, dtype=q.dtype)
        tolerance = singular[0] * max(self.basis.shape) * torch.finfo(singular.dtype).eps
        return {
            "basis_rank": float((singular > tolerance).sum().item()),
            "basis_min_singular_value": float(singular[-1].item()),
            "basis_condition": float((singular[0] / singular[-1].clamp_min(1e-12)).item()),
            "q_orth_error": float((q.T @ q - eye).square().mean().item()),
        }


class StableFeatureProjection(nn.Module):
    """Fixed-beta projection with no LayerNorm, learned gate, or task gradient to B."""
    def __init__(self, feature_dim, rank=8, beta=1.0, seed=0):
        super().__init__()
        if not math.isfinite(float(beta)) or not 0 <= float(beta) <= 1:
            raise ValueError("DICM beta must be in [0,1]")
        self.subspace = InterventionSensitiveSubspace(feature_dim, rank, seed)
        self.register_buffer("beta", torch.tensor(float(beta)))

    def decompose(self, z, q=None):
        sensitive = self.subspace.project(z, q=q, detach_basis=True)
        stable = z - self.beta.to(z.dtype) * sensitive
        return stable, sensitive

    def forward(self, z, q=None):
        if self.beta.item() == 0:
            return z
        return self.decompose(z, q=q)[0]
