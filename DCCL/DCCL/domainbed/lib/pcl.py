import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCLLoss(nn.Module):
    """PCL-style auxiliary loss for class-conditioned cross-domain alignment.

    The module is parameter-free. It computes a partial optimal-transport plan
    for each (class, domain-pair) subset and uses that plan only as a local
    alignment signal. It never changes the positive mask used by DCCL SupCon.
    """

    def __init__(
        self,
        transport_mass=0.8,
        sinkhorn_epsilon=0.05,
        sinkhorn_iters=50,
        uniform_temperature=2.0,
        match_threshold=0.5,
        eps=1e-8,
    ):
        super().__init__()
        if not 0.0 < transport_mass <= 1.0:
            raise ValueError("transport_mass must be in (0, 1].")
        if sinkhorn_epsilon <= 0:
            raise ValueError("sinkhorn_epsilon must be > 0.")
        if sinkhorn_iters <= 0:
            raise ValueError("sinkhorn_iters must be > 0.")
        if uniform_temperature <= 0:
            raise ValueError("uniform_temperature must be > 0.")
        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError("match_threshold must be in [0, 1].")

        self.transport_mass = float(transport_mass)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.uniform_temperature = float(uniform_temperature)
        self.match_threshold = float(match_threshold)
        self.eps = float(eps)

    @staticmethod
    def cosine_cost(z_a, z_b):
        """Cosine-distance cost on normalized representations."""
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)
        return 1.0 - torch.matmul(z_a, z_b.t())

    def _masked_partial_sinkhorn(self, cost, valid_mask=None):
        """Entropic partial OT via balanced OT with one dummy node per side.

        Real-to-real transported mass is constrained to ``transport_mass``.
        Real-to-dummy and dummy-to-real edges absorb the unmatched mass.
        The dummy-to-dummy edge is forbidden so unmatched mass cannot bypass
        the requested real-to-real transport amount.
        """
        if cost.ndim != 2:
            raise ValueError("cost must be a matrix.")
        n, m = cost.shape
        if n == 0 or m == 0:
            return cost.new_zeros((n, m))

        if valid_mask is None:
            valid_mask = torch.ones_like(cost, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=cost.device, dtype=torch.bool)
            if valid_mask.shape != cost.shape:
                raise ValueError("valid_mask must have the same shape as cost.")

        # In DCCL this solver is called after class/domain subsetting, so the
        # class-conditioned mask is normally all True. Keep the mask in the
        # solver to preserve the masked m-POT interface.
        kernel_real = torch.exp(-cost / self.sinkhorn_epsilon)
        kernel_real = kernel_real * valid_mask.to(kernel_real.dtype)

        kernel = cost.new_zeros((n + 1, m + 1))
        kernel[:n, :m] = kernel_real
        # Zero-cost dummy edges absorb unmatched mass.
        kernel[:n, m] = 1.0
        kernel[n, :m] = 1.0
        # Forbid dummy -> dummy, otherwise the requested partial mass is not
        # identifiable.
        kernel[n, m] = 0.0

        unmatched = 1.0 - self.transport_mass
        a = cost.new_full((n + 1,), 1.0 / n)
        b = cost.new_full((m + 1,), 1.0 / m)
        a[n] = unmatched
        b[m] = unmatched

        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(self.sinkhorn_iters):
            kv = torch.matmul(kernel, v)
            u = a / kv.clamp_min(self.eps)
            ktu = torch.matmul(kernel.t(), u)
            v = b / ktu.clamp_min(self.eps)

        plan = u[:, None] * kernel * v[None, :]
        return plan[:n, :m]

    def _uniformity(self, features):
        """Hyperspherical uniformity loss (lower is more uniform)."""
        if features.shape[0] < 2:
            return features.sum() * 0.0
        z = F.normalize(features, dim=1)
        sq_dist = torch.pdist(z, p=2).pow(2)
        return torch.log(
            torch.exp(-self.uniform_temperature * sq_dist).mean().clamp_min(self.eps)
        )

    def forward(self, features, labels, domains):
        if features.ndim != 2:
            raise ValueError("features must have shape [N, D].")
        labels = labels.reshape(-1).to(features.device)
        domains = domains.reshape(-1).to(features.device)
        if not (features.shape[0] == labels.numel() == domains.numel()):
            raise ValueError("features, labels and domains must contain the same N samples.")

        zero = features.sum() * 0.0
        align_sum = zero
        valid_class_pairs = 0
        valid_domain_pairs = 0
        transported_mass_sum = 0.0
        matching_ratio_sum = 0.0

        unique_domains = torch.unique(domains, sorted=True).tolist()
        unique_labels = torch.unique(labels, sorted=True).tolist()

        for domain_a, domain_b in itertools.combinations(unique_domains, 2):
            domain_has_pair = False
            mask_a_domain = domains == domain_a
            mask_b_domain = domains == domain_b

            for class_id in unique_labels:
                idx_a = torch.nonzero(
                    mask_a_domain & (labels == class_id), as_tuple=False
                ).flatten()
                idx_b = torch.nonzero(
                    mask_b_domain & (labels == class_id), as_tuple=False
                ).flatten()
                if idx_a.numel() == 0 or idx_b.numel() == 0:
                    continue

                z_a = features.index_select(0, idx_a)
                z_b = features.index_select(0, idx_b)
                cost = self.cosine_cost(z_a, z_b)

                # Labels and different-domain constraints were already used to
                # form this submatrix, therefore every cell is a legal
                # same-class cross-domain candidate.
                valid_mask = torch.ones_like(cost, dtype=torch.bool)
                with torch.no_grad():
                    transport = self._masked_partial_sinkhorn(
                        cost.detach(), valid_mask
                    )

                mass = transport.sum()
                if not torch.isfinite(mass) or mass.item() <= self.eps:
                    continue

                pair_align = (transport * cost).sum() / mass.clamp_min(self.eps)
                if not torch.isfinite(pair_align):
                    continue

                align_sum = align_sum + pair_align
                valid_class_pairs += 1
                domain_has_pair = True
                transported_mass_sum += float(mass.detach().item())

                # Diagnostic only: count cells carrying at least a fraction of
                # the strongest transport in their row.
                row_max = transport.max(dim=1, keepdim=True).values
                active = (row_max > self.eps) & (
                    transport
                    >= self.match_threshold * row_max.clamp_min(self.eps)
                )
                matching_ratio_sum += float(active.float().mean().item())

            if domain_has_pair:
                valid_domain_pairs += 1

        if valid_class_pairs > 0:
            align_loss = align_sum / valid_class_pairs
            avg_transport_mass = transported_mass_sum / valid_class_pairs
            avg_matching_ratio = matching_ratio_sum / valid_class_pairs
        else:
            align_loss = zero
            avg_transport_mass = 0.0
            avg_matching_ratio = 0.0

        uniform_loss = self._uniformity(features)

        stats = {
            "valid_domain_pairs": float(valid_domain_pairs),
            "valid_class_pairs": float(valid_class_pairs),
            "transport_mass": float(avg_transport_mass),
            "matching_ratio": float(avg_matching_ratio),
        }
        return align_loss, uniform_loss, stats
