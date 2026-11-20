"""Source-trained directional attenuation in the current QR subspace."""

from typing import NamedTuple

import torch
from torch import nn


MIN_RELATIVE_REMOVAL = 0.01
MIN_GAIN_SPREAD = 1e-3


class GateError(RuntimeError):
    """A gate estimate without enough signal to continue the fixed protocol."""


class DirectionEstimate(NamedTuple):
    rotation: torch.Tensor
    gains: torch.Tensor
    class_scatter: torch.Tensor
    domain_scatter: torch.Tensor
    class_scores: torch.Tensor
    domain_scores: torch.Tensor
    direction_scores: torch.Tensor
    traces: torch.Tensor
    supported_classes: int
    relative_removal: torch.Tensor


@torch.no_grad()
def estimate_direction_gate(q, prototypes, valid, observations):
    """Use all-source-supported, equally weighted class/domain centroids.

    Normalize the two scatter matrices to unit trace before diagonalizing
    their difference. This compares directional concentration, independent
    of the overall scale of class versus source-domain variation.
    """
    if q.ndim != 2 or prototypes.ndim != 3 or prototypes.shape[-1] != q.shape[0]:
        raise GateError("Invalid shape for directional gate prototypes or QR basis")
    if valid.shape != prototypes.shape[:2] or valid.dtype != torch.bool:
        raise GateError("Invalid gate prototype support mask")
    if observations.ndim != 2 or observations.shape[1] != q.shape[0] or not len(observations):
        raise GateError("Gate energy check requires nonempty source observations")
    supported = valid.all(dim=1)
    n_classes = int(supported.sum().item())
    if n_classes < 2:
        raise GateError(f"Gate requires two classes supported by all source domains, found {n_classes}")

    q = q.detach().double()
    centers = prototypes[supported].detach().double() @ q
    if not torch.isfinite(centers).all():
        raise GateError("Non-finite projected source-domain prototypes")

    class_centers = centers.mean(dim=1)
    class_offsets = class_centers - class_centers.mean(dim=0)
    domain_offsets = (centers - class_centers[:, None, :]).flatten(0, 1)
    class_scatter = class_offsets.T @ class_offsets / len(class_offsets)
    domain_scatter = domain_offsets.T @ domain_offsets / len(domain_offsets)
    traces = torch.stack((class_scatter.trace(), domain_scatter.trace()))
    if not torch.isfinite(traces).all() or (traces <= 1e-12).any():
        raise GateError(f"Gate scatter has no usable class or domain energy: {traces.tolist()}")

    class_scatter = class_scatter / traces[0]
    domain_scatter = domain_scatter / traces[1]
    directions, rotation = torch.linalg.eigh(domain_scatter - class_scatter)
    class_scores = (rotation * (class_scatter @ rotation)).sum(dim=0).clamp_min(0)
    domain_scores = (rotation * (domain_scatter @ rotation)).sum(dim=0).clamp_min(0)
    total = class_scores + domain_scores
    gains = torch.where(total > 1e-12, domain_scores / total.clamp_min(1e-12), 0.0)
    if not torch.isfinite(gains).all() or not torch.isfinite(directions).all():
        raise GateError("Non-finite directional gate coefficients")
    if (gains.min() < 0 or gains.max() > 1 or gains.max() - gains.min() <= MIN_GAIN_SPREAD):
        raise GateError(f"Degenerate gate coefficients: min={gains.min().item()}, max={gains.max().item()}")

    coordinates = (observations.detach().double() @ q) @ rotation
    uniform_energy = coordinates.square().sum()
    if not torch.isfinite(uniform_energy) or uniform_energy <= 1e-12:
        raise GateError("Gate energy check has no projected source energy")
    relative_removal = (coordinates * gains).square().sum() / uniform_energy
    if not torch.isfinite(relative_removal) or relative_removal < MIN_RELATIVE_REMOVAL:
        raise GateError(f"Gate removed only {relative_removal.item():.5g} of uniform projected energy")

    return DirectionEstimate(rotation, gains, class_scatter, domain_scatter,
                             class_scores, domain_scores, directions, traces,
                             n_classes, relative_removal)


class DirectionGate(nn.Module):
    """Persistent source-only EMA-derived gate, shared by every example."""

    def __init__(self, feature_dim, rank):
        super().__init__()
        self.register_buffer("rotation", torch.eye(rank))
        self.register_buffer("gains", torch.ones(rank))
        self.register_buffer("class_scatter", torch.zeros(rank, rank))
        self.register_buffer("domain_scatter", torch.zeros(rank, rank))
        self.register_buffer("class_scores", torch.zeros(rank))
        self.register_buffer("domain_scores", torch.zeros(rank))
        self.register_buffer("direction_scores", torch.zeros(rank))
        self.register_buffer("traces", torch.zeros(2, dtype=torch.float64))
        self.register_buffer("anchor", torch.zeros(feature_dim, rank))
        self.register_buffer("last_step", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("supported_classes", torch.tensor(0, dtype=torch.long))
        self.register_buffer("relative_removal", torch.tensor(1.))
        self.register_buffer("operator_drift", torch.tensor(0.))

    @torch.no_grad()
    def refresh(self, q, prototypes, valid, observations, step):
        estimate = estimate_direction_gate(q, prototypes, valid, observations)
        basis = q.detach().double() @ estimate.rotation
        drift = 0.
        if self.last_step.item() >= 0:
            previous = self.gains.double()
            overlap = (basis.T @ self.anchor.double()).square()
            inner = estimate.gains @ overlap @ previous
            distance2 = estimate.gains.square().sum() + previous.square().sum() - 2 * inner
            drift = float(distance2.clamp_min(0).sqrt().item())
        for name in DirectionEstimate._fields:
            value = getattr(estimate, name)
            buffer = getattr(self, name)
            if isinstance(value, torch.Tensor):
                buffer.copy_(value)
            else:
                buffer.fill_(value)
        self.anchor.copy_(basis)
        self.last_step.fill_(step)
        self.operator_drift.fill_(drift)

    def basis(self, q):
        return q.detach().float() @ self.rotation

    def suppressible(self, z, q):
        with torch.autocast(device_type=z.device.type, enabled=False):
            v = self.basis(q)
            return (((z.float() @ v) * self.gains) @ v.T).to(z.dtype)

    def effective_dimension(self):
        squared = self.gains.square()
        return squared.sum().square() / squared.square().sum().clamp_min(1e-12)

    def snapshot(self):
        return dict(step=int(self.last_step.item()), gains=self.gains.tolist(),
                    class_scores=self.class_scores.tolist(), domain_scores=self.domain_scores.tolist(),
                    direction_scores=self.direction_scores.tolist(), traces=self.traces.tolist(),
                    effective_dimension=float(self.effective_dimension().item()),
                    directions_at_least_half=int((self.gains >= .5).sum().item()),
                    relative_removal=float(self.relative_removal.item()),
                    operator_drift=float(self.operator_drift.item()),
                    supported_classes=int(self.supported_classes.item()))
