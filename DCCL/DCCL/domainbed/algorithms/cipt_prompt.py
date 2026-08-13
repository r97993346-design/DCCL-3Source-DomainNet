"""CLIP loading, prompt learning, and selectable CIPT textual contexts."""

from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

# The repository vendors the official OpenAI CLIP package. Import it directly
# so a local checkpoint works in an offline environment without a pip install.
_BUNDLED_CLIP = Path(__file__).resolve().parents[4] / "CLIP"
if str(_BUNDLED_CLIP) not in sys.path:
    sys.path.insert(0, str(_BUNDLED_CLIP))
import clip


# B5a: original high-performance feature/multiprompt prompts. Keep this exactly
# as the default so the ablation branch starts from the known strong baseline.
B5A_GENERIC_TEMPLATES = (
    "a low quality photo.",
    "a high quality photo.",
    "a photo in an unusual style.",
    "a photo in an unusual context.",
)


# B5b: OpenAI ImageNet prompt bank used by the official CIPT implementation.
# These prompts are class-conditioned via the {} placeholder.
B5B_IMAGENET_TEMPLATES = (
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
)


# B5c: expanded class-agnostic bank inspired by B5b. It deliberately removes
# class names so interventions describe image quality/style/context only.
B5C_GENERIC_EXPANDED_TEMPLATES = (
    "a bad photo.",
    "a photo with many objects.",
    "a sculpture.",
    "a hard to see photo.",
    "a low resolution photo.",
    "a rendering.",
    "graffiti.",
    "a cropped photo.",
    "a tattoo.",
    "an embroidered image.",
    "a bright photo.",
    "a clean photo.",
    "a dirty photo.",
    "a dark photo.",
    "a drawing.",
    "a plastic object.",
    "a close-up photo.",
    "a black and white photo.",
    "a painting.",
    "a pixelated photo.",
    "a jpeg corrupted photo.",
    "a blurry photo.",
    "a natural photo.",
    "a good photo.",
    "a video game rendering.",
    "a doodle.",
    "an origami object.",
    "a toy.",
    "a rendition.",
    "a nice photo.",
    "a weird photo.",
    "a cartoon.",
    "artwork.",
    "a sketch.",
    "a plushie.",
    "a low quality photo.",
    "a high quality photo.",
    "an image in an unusual style.",
    "an image in an unusual context.",
    "an image with unusual texture.",
    "an image with unusual lighting.",
    "an image with an unusual background.",
)

TEMPLATE_MODES = ("b5a", "b5b", "b5c")


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
    """Learnable class prompts plus three selectable intervention template banks."""

    def __init__(self, class_names, clip_model, tokenize, prompt_length, prompt_init, k):
        super().__init__()
        self.clip_model = clip_model
        self.text_encoder = CLIPTextEncoder(clip_model)
        self.prompt_learner = PromptLearner(class_names, clip_model, tokenize, prompt_length, prompt_init)
        self.class_names = [name.replace("_", " ") for name in class_names]
        self.k = int(k)
        self.template_mode = "b5a"

        self.register_buffer(
            "b5a_text_bank",
            self._encode_texts(list(B5A_GENERIC_TEMPLATES), tokenize),
        )
        self.register_buffer(
            "b5c_text_bank",
            self._encode_texts(list(B5C_GENERIC_EXPANDED_TEMPLATES), tokenize),
        )

        # [num_classes, num_templates, dim]
        b5b_texts = [
            template.format(class_name)
            for class_name in self.class_names
            for template in B5B_IMAGENET_TEMPLATES
        ]
        b5b_encoded = self._encode_texts(b5b_texts, tokenize)
        self.register_buffer(
            "b5b_text_bank",
            b5b_encoded.reshape(
                len(self.class_names), len(B5B_IMAGENET_TEMPLATES), -1
            ),
        )

    def _encode_texts(self, texts, tokenize, batch_size=256):
        encoded_chunks = []
        device = next(self.clip_model.parameters()).device
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                tokens = tokenize(texts[start : start + batch_size]).to(device)
                encoded = self.clip_model.encode_text(tokens).float()
                encoded_chunks.append(F.normalize(encoded, dim=-1))
        return torch.cat(encoded_chunks, dim=0)

    def set_template_mode(self, mode):
        mode = str(mode).lower()
        if mode not in TEMPLATE_MODES:
            raise ValueError(
                "Unknown cipt_template_mode={!r}; expected one of {}".format(
                    mode, TEMPLATE_MODES
                )
            )
        self.template_mode = mode

    def _select_indices(self, num_available, device, legacy_fixed=False):
        if num_available < 1:
            raise ValueError("Template bank must contain at least one prompt.")
        if legacy_fixed:
            return torch.arange(self.k, device=device) % num_available
        if self.training:
            if self.k <= num_available:
                return torch.randperm(num_available, device=device)[: self.k]
            pieces = []
            remaining = self.k
            while remaining > 0:
                perm = torch.randperm(num_available, device=device)
                take = min(remaining, num_available)
                pieces.append(perm[:take])
                remaining -= take
            return torch.cat(pieces, dim=0)
        return torch.arange(self.k, device=device) % num_available

    def intervention_features(self, labels=None):
        """Return selected intervention embeddings for the active B5 mode.

        B5a -> [K, D], exact legacy fixed/cycled prompts.
        B5c -> [K, D], random K during training and deterministic K at eval.
        B5b with labels -> [B, K, D], class-conditioned official prompts.
        B5b without labels -> [C, K, D], used for candidate-class inference.
        """
        if self.template_mode == "b5a":
            bank = self.b5a_text_bank
            idx = self._select_indices(bank.shape[0], bank.device, legacy_fixed=True)
            return bank.index_select(0, idx)

        if self.template_mode == "b5c":
            bank = self.b5c_text_bank
            idx = self._select_indices(bank.shape[0], bank.device)
            return bank.index_select(0, idx)

        bank = self.b5b_text_bank
        idx = self._select_indices(bank.shape[1], bank.device)
        selected = bank.index_select(1, idx)
        if labels is None:
            return selected
        return selected[labels.to(device=bank.device, dtype=torch.long)]

    @property
    def irrelevant_text_features(self):
        # Backward-compatible path used by the original high-performance
        # CIPTDCCL implementation. B5b is handled by the ablation wrapper,
        # because class-conditioned prompts need labels/candidate classes.
        return self.intervention_features(labels=None)

    def class_features(self):
        features = self.text_encoder(self.prompt_learner(), self.prompt_learner.tokenized_prompts)
        return F.normalize(features.float(), dim=-1)
