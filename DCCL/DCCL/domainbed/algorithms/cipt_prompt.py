"""CLIP loading, prompt learning, and CIPT textual contexts."""

from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

# The repository vendors the official OpenAI CLIP package.  Import it directly
# so a local checkpoint works in an offline environment without a pip install.
_BUNDLED_CLIP = Path(__file__).resolve().parents[4] / "CLIP"
if str(_BUNDLED_CLIP) not in sys.path:
    sys.path.insert(0, str(_BUNDLED_CLIP))
import clip


IRRELEVANT_TEMPLATES = (
    "a low quality photo.",
    "a high quality photo.",
    "a photo in an unusual style.",
    "a photo in an unusual context.",
)


def load_frozen_clip(backbone, local_path=""):
    """Load OpenAI CLIP, preferring an explicit local checkpoint without network I/O."""
    if local_path:
        path = Path(local_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError("CIPT offline CLIP checkpoint does not exist: {}".format(path))
        model, _ = clip.load(str(path), device="cpu", jit=False)
    else:
        model, _ = clip.load(backbone, device="cpu", jit=False)
    model.float()
    model.requires_grad_(False)
    return model, clip.tokenize


class CLIPTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, prompts, tokenized_prompts):
        model = self.clip_model
        x = prompts + model.positional_embedding.to(prompts.dtype)
        x = x.permute(1, 0, 2)
        x = model.transformer(x)
        x = x.permute(1, 0, 2)
        x = model.ln_final(x).to(prompts.dtype)
        return x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ model.text_projection


class PromptLearner(nn.Module):
    """CoOp-style learnable CLIP context tokens used by CIPT."""

    def __init__(self, class_names, clip_model, tokenize, prompt_length=16, prompt_init="a photo of a"):
        super().__init__()
        dtype = clip_model.dtype
        width = clip_model.ln_final.weight.shape[0]
        init_words = prompt_init.replace("_", " ").split()
        if prompt_init and len(init_words) <= prompt_length:
            initialized = tokenize(prompt_init)
            with torch.no_grad():
                init_embedding = clip_model.token_embedding(initialized).to(dtype)
            context = init_embedding[0, 1 : 1 + len(init_words)]
            if len(init_words) < prompt_length:
                context = torch.cat([context, torch.empty(prompt_length-len(init_words), width, dtype=dtype).normal_(std=0.02)])
        else:
            context = torch.empty(prompt_length, width, dtype=dtype).normal_(std=0.02)
        self.context = nn.Parameter(context)
        names = [name.replace("_", " ") for name in class_names]
        tokenized = tokenize(["X " * prompt_length + name + "." for name in names])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).to(dtype)
        self.register_buffer("token_prefix", embedding[:, :1])
        self.register_buffer("token_suffix", embedding[:, 1 + prompt_length :])
        self.register_buffer("tokenized_prompts", tokenized)

    def forward(self):
        context = self.context.unsqueeze(0).expand(self.token_prefix.shape[0], -1, -1)
        return torch.cat((self.token_prefix, context, self.token_suffix), dim=1)


class CIPTTextFeatures(nn.Module):
    def __init__(self, class_names, clip_model, tokenize, prompt_length, prompt_init, k):
        super().__init__()
        self.clip_model = clip_model
        self.text_encoder = CLIPTextEncoder(clip_model)
        self.prompt_learner = PromptLearner(class_names, clip_model, tokenize, prompt_length, prompt_init)
        templates = [IRRELEVANT_TEMPLATES[i % len(IRRELEVANT_TEMPLATES)] for i in range(k)]
        irrelevant_tokens = tokenize(templates)
        with torch.no_grad():
            irrelevant = clip_model.encode_text(irrelevant_tokens).float()
        self.register_buffer("irrelevant_text_features", F.normalize(irrelevant, dim=-1))

    def class_features(self):
        features = self.text_encoder(self.prompt_learner(), self.prompt_learner.tokenized_prompts)
        return F.normalize(features.float(), dim=-1)
