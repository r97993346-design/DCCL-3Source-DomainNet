"""CIRL internal causal enhancement for DCCL.

This module is intentionally lightweight and opt-in.  It lives inside the
student training path (between DCCL projection and contrastive learning) and
produces a pair reliability matrix that can reweight supervised contrastive
relations without changing the default DCCL baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIRLModule(nn.Module):
    """Causal reference reweighting and reliability regularization.

    The reliability matrix favors same-class cross-domain connections and
    suppresses visually similar cross-class pairs.  The optional Fourier branch
    estimates style/frequency stability directly from projected features.
    """

    def __init__(
        self,
        temperature=1.0,
        min_reliability=0.05,
        use_fourier_reliability=False,
        fourier_alpha=0.5,
    ):
        super().__init__()
        self.temperature = max(float(temperature), 1e-6)
        self.min_reliability = float(min_reliability)
        self.use_fourier_reliability = bool(use_fourier_reliability)
        self.fourier_alpha = float(fourier_alpha)

    def _fourier_style_reliability(self, z):
        spectrum = torch.fft.rfft(z.float(), dim=1)
        amplitude = spectrum.abs()
        if amplitude.size(1) <= 1:
            return torch.ones(z.size(0), z.size(0), device=z.device, dtype=z.dtype)
        split = max(1, amplitude.size(1) // 4)
        low_freq = amplitude[:, :split]
        high_freq = amplitude[:, split:]
        low_sim = F.normalize(low_freq, dim=1) @ F.normalize(low_freq, dim=1).T
        if high_freq.size(1) == 0:
            style_rel = low_sim
        else:
            high_sim = F.normalize(high_freq, dim=1) @ F.normalize(high_freq, dim=1).T
            # Reward shared low-frequency semantic structure while damping pairs
            # connected mostly by high-frequency texture/style evidence.
            style_rel = torch.sigmoid((low_sim - high_sim) / self.temperature)
        return style_rel.to(dtype=z.dtype)


    def fourier_intervention(self, images, alpha=None):
        """Create a Fourier amplitude intervention batch for ICR-style checks.

        The intervention swaps low-frequency amplitude statistics across samples
        while preserving each image phase. It is used only when CIRL Fourier
        reliability is enabled, so baseline DCCL never pays this cost.
        """
        if images.size(0) <= 1:
            return images
        alpha = self.fourier_alpha if alpha is None else float(alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        fft = torch.fft.fft2(images.float(), dim=(-2, -1))
        amplitude, phase = torch.abs(fft), torch.angle(fft)
        mixed_amplitude = amplitude.roll(shifts=1, dims=0)
        h, w = images.shape[-2:]
        mask = torch.zeros((h, w), device=images.device, dtype=images.dtype)
        radius_h = max(1, int(h * 0.125))
        radius_w = max(1, int(w * 0.125))
        center_h, center_w = h // 2, w // 2
        mask[center_h - radius_h:center_h + radius_h, center_w - radius_w:center_w + radius_w] = 1.0
        mask = torch.fft.ifftshift(mask).view(1, 1, h, w)
        intervened_amp = amplitude * (1.0 - alpha * mask) + mixed_amplitude * (alpha * mask)
        intervened = torch.fft.ifft2(intervened_amp * torch.exp(1j * phase), dim=(-2, -1)).real
        return intervened.to(dtype=images.dtype)

    def forward(self, z, y, domain):
        z_norm = F.normalize(z, dim=1)
        sim = z_norm @ z_norm.T
        y = y.view(-1)
        domain = domain.view(-1)
        same_class = torch.eq(y[:, None], y[None, :]).float()
        same_domain = torch.eq(domain[:, None], domain[None, :]).float()
        cross_domain = 1.0 - same_domain
        eye = torch.eye(z.size(0), device=z.device, dtype=z.dtype)

        semantic_rel = torch.sigmoid(sim / self.temperature)
        causal_prior = 0.25 + 0.75 * same_class
        cross_domain_boost = 1.0 + 0.25 * same_class * cross_domain
        cross_class_penalty = 1.0 - 0.50 * (1.0 - same_class) * semantic_rel
        reliability = semantic_rel * causal_prior * cross_domain_boost * cross_class_penalty

        if self.use_fourier_reliability:
            style_rel = self._fourier_style_reliability(z_norm)
            alpha = min(max(self.fourier_alpha, 0.0), 1.0)
            reliability = (1.0 - alpha) * reliability + alpha * reliability * style_rel

        reliability = reliability.clamp(min=self.min_reliability, max=1.0)
        reliability = reliability * (1.0 - eye) + eye

        denom = (reliability * same_class * (1.0 - eye)).sum().clamp_min(1.0)
        loss_reliability = ((1.0 - reliability) * same_class * (1.0 - eye)).sum() / denom

        pos_mask = same_class * cross_domain * (1.0 - eye)
        if pos_mask.sum() > 0:
            causal_center = (reliability * pos_mask) @ z_norm
            causal_center = causal_center / (reliability * pos_mask).sum(dim=1, keepdim=True).clamp_min(1.0)
            valid = (pos_mask.sum(dim=1) > 0).float()
            loss_consistency = ((1.0 - F.cosine_similarity(z_norm, causal_center, dim=1)) * valid).sum() / valid.sum().clamp_min(1.0)
            z_causal = F.normalize(z + causal_center, dim=1)
        else:
            loss_consistency = z_norm.new_tensor(0.0)
            z_causal = z_norm

        loss_cirl = loss_reliability + loss_consistency
        return {
            "z_causal": z_causal,
            "reliability_matrix": reliability.detach(),
            "loss_cirl": loss_cirl,
            "loss_reliability": loss_reliability,
            "loss_consistency": loss_consistency,
            "mean_reliability": reliability.detach().mean(),
            "min_reliability": reliability.detach().min(),
            "max_reliability": reliability.detach().max(),
        }
