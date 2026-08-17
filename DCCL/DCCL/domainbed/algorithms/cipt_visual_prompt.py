"""Shallow visual prompt tuning for the frozen OpenAI CLIP ViT backbone.

CLIP parameters stay frozen. Gradients flow through the frozen transformer only
so the learnable visual prompt tokens can be optimized. The prompt is inserted
once, before the vision transformer, and shared by original/augmented views.
"""

import torch
from torch import nn


class VisualPromptLearner(nn.Module):
    """A small set of learnable tokens inserted into CLIP's visual sequence."""

    def __init__(self, clip_model, prompt_length=4):
        super().__init__()
        visual = clip_model.visual
        required = ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post")
        if not all(hasattr(visual, name) for name in required):
            raise TypeError("Visual prompt tuning currently requires a CLIP ViT visual backbone.")
        if prompt_length < 1:
            raise ValueError("cipt_visual_prompt_length must be >= 1")

        width = int(visual.conv1.out_channels)
        dtype = visual.conv1.weight.dtype
        context = torch.empty(int(prompt_length), width, dtype=dtype)
        nn.init.normal_(context, std=0.02)
        self.context = nn.Parameter(context)
        self.prompt_length = int(prompt_length)

    def forward(self, batch_size, dtype, device):
        return self.context.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)


def encode_image_with_visual_prompt(clip_model, images, prompt_learner):
    """Encode images with a shallow visual prompt while keeping CLIP frozen.

    Important: this function intentionally does NOT use torch.no_grad(). CLIP's
    own parameters have requires_grad=False, but autograd must retain the path
    through the frozen transformer so gradients can reach prompt_learner.context.
    """
    visual = clip_model.visual
    x = images.to(dtype=visual.conv1.weight.dtype)
    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)

    prompts = prompt_learner(x.shape[0], x.dtype, x.device)
    # Keep CLIP's pretrained positional embeddings unchanged for CLS/patches;
    # visual prompt tokens are extra learnable context tokens with no fixed pos.
    x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x[:, 0, :])

    if visual.proj is not None:
        x = x @ visual.proj
    return x.float()
