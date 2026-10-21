"""Label-free per-image random sampling from the complete official B5b bank."""

import numbers

import torch
from torch import nn


class RandomPromptSelector(nn.Module):
    """Uniformly sample n distinct template positions independently per image.

    Feature values, labels, E/S and CLIP logits never enter sampling. One index
    identifies the same template position for every class. Fresh permutations
    use the seeded PyTorch RNG at every call; normal evaluation never calls it.
    """

    def __init__(self, topk=40):
        super().__init__()
        if isinstance(topk, bool) or not isinstance(topk, numbers.Integral) or not 2 <= topk <= 80:
            raise ValueError("cipt_env_topk must be an integer between 2 and 80")
        self.topk = int(topk)

    @torch.no_grad()
    def forward(self, visual_features, text_bank):
        # Only shape/device are needed. Do not normalize features, compute
        # logits/softmax/entropy, or invoke the inherited batch-shared selector.
        if visual_features.ndim != 2 or min(visual_features.shape) < 1:
            raise ValueError("visual_features must have nonempty shape [B,D]")
        if (text_bank.ndim != 3 or text_bank.shape[0] < 1
                or text_bank.shape[1:] != (80, visual_features.shape[1])):
            raise ValueError("text_bank must have shape [C,80,D] matching features")
        if visual_features.device != text_bank.device:
            raise ValueError("visual_features and text_bank must be on the same device")
        indices = torch.stack([
            torch.randperm(80, device=visual_features.device)[:self.topk]
            for _ in range(visual_features.shape[0])
        ])
        active = torch.ones((), dtype=torch.float32, device=visual_features.device)
        return indices, {"entropy_selector_active": active.new_zeros(()),
                         "random_selector_active": active}
