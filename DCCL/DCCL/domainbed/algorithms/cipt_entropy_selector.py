"""Label-free selection of official B5b templates using frozen CLIP features."""

from numbers import Integral

import torch
from torch import nn
import torch.nn.functional as F


class EntropyPromptSelector(nn.Module):
    """Select the lowest-entropy N templates separately for each image.

    Each template defines a classifier over ALL classes. The selected template
    indices are shared across candidate classes; neither labels nor E/S enter
    selection. Ties retain the official template order for determinism.
    """

    def __init__(self, topk=40):
        super().__init__()
        if isinstance(topk, bool) or not isinstance(topk, Integral) or not 1 <= topk <= 80:
            raise ValueError("cipt_entropy_topk must be an integer between 1 and 80")
        self.topk = int(topk)

    @torch.no_grad()
    def forward(self, visual_features, text_bank, logit_scale):
        if visual_features.ndim != 2 or visual_features.shape[0] < 1:
            raise ValueError("Entropy selection expects visual features [B,D], B >= 1")
        if (text_bank.ndim != 3 or text_bank.shape[0] < 2
                or text_bank.shape[1] != 80
                or text_bank.shape[2] != visual_features.shape[1]
                or text_bank.shape[2] < 1):
            raise ValueError("Entropy selection expects the full B5b bank [C,80,D], C >= 2")
        if visual_features.device != text_bank.device:
            raise ValueError("Visual features and the B5b bank must share a device")
        scale = torch.as_tensor(logit_scale, device=visual_features.device).detach().float()
        if scale.numel() != 1 or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("CLIP logit scale must be a finite positive scalar")
        visual = F.normalize(visual_features.detach().float(), dim=-1)
        text = F.normalize(text_bank.detach().float(), dim=-1)
        logits = scale * torch.einsum("bd,cnd->bnc", visual, text)
        log_prob = F.log_softmax(logits.float(), dim=-1)
        entropy = -(log_prob.exp() * log_prob).sum(dim=-1)
        # Stable sorting makes an all-tied bank choose 0,1,... rather than an
        # implementation-dependent subset. There are only 80 templates.
        indices = entropy.argsort(dim=1, stable=True)[:, :self.topk]
        selected_entropy = entropy.gather(1, indices)
        return indices, {
            "entropy_selector_active": entropy.new_ones(()),
            "env_entropy_mean": entropy.mean(),
            "env_selected_entropy_mean": selected_entropy.mean(),
            "env_selected_entropy_min": selected_entropy.min(),
            "env_selected_entropy_max": selected_entropy.max(),
        }
