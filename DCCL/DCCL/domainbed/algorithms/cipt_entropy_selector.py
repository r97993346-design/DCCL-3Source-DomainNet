"""Label-free, per-image selection from the complete official B5b bank."""

import numbers

import torch
from torch import nn
import torch.nn.functional as F


def class_prompt_logits(features, text_bank, logit_scale):
    """Cosine logits [B,80,C]; ``logit_scale`` is CLIP's exp(logit_scale).

    The feature argument keeps its gradient (E for MCC, frozen V for selection).
    Official text vectors and the frozen scale never receive gradients.
    """
    if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("features must have nonempty shape [B,D]")
    if (text_bank.ndim != 3 or text_bank.shape[0] < 1
            or text_bank.shape[1] != 80 or text_bank.shape[2] != features.shape[1]):
        raise ValueError("text_bank must have shape [C,80,D] matching features")
    if features.device != text_bank.device:
        raise ValueError("features and text_bank must be on the same device")
    scale = torch.as_tensor(logit_scale, device=features.device).detach().float()
    if scale.numel() != 1 or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("CLIP logit scale must be a finite positive scalar")
    unit_features = F.normalize(features.float(), dim=-1)
    unit_text = F.normalize(text_bank.detach().float(), dim=-1)
    return scale * torch.einsum("bd,cjd->bjc", unit_features, unit_text).float()


class EntropyPromptSelector(nn.Module):
    """Choose each image's lowest-entropy templates across ALL classes.

    No label argument is accepted. One selected index identifies the same
    template for every class. Selection uses frozen CLIP visual features,
    never the trainable causal/spurious adapters.
    """

    def __init__(self, topk=40):
        super().__init__()
        if isinstance(topk, bool) or not isinstance(topk, numbers.Integral) or not 2 <= topk <= 80:
            raise ValueError("cipt_env_topk must be an integer between 2 and 80")
        self.topk = int(topk)

    @torch.no_grad()
    def forward(self, visual_features, text_bank, logit_scale):
        logits = class_prompt_logits(visual_features, text_bank, logit_scale)
        log_prob = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        entropy = -(log_prob.exp() * log_prob).sum(dim=-1)
        selected_entropy, indices = torch.topk(
            entropy, k=self.topk, dim=1, largest=False
        )
        return indices, {
            "entropy_selector_active": entropy.new_ones(()),
            "env_entropy_mean": entropy.mean(),
            "env_selected_entropy_mean": selected_entropy.mean(),
            "env_selected_entropy_min": selected_entropy.min(),
            "env_selected_entropy_max": selected_entropy.max(),
        }
