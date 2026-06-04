"""RISE-guided CLIP supervision utilities for DCCL.

This module keeps CLIP teacher construction and losses isolated from the
baseline DCCL path. It is imported lazily by DCCL only when --use_rise is set.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


PROMPT_TEMPLATES = {
    "simple": [
        "a photo of a {class_name}",
    ],
    "multi": [
        "a photo of a {class_name}",
        "a sketch of a {class_name}",
        "a painting of a {class_name}",
        "a clipart image of a {class_name}",
        "a drawing of a {class_name}",
    ],
    "domain_invariant": [
        "a photo of a {class_name}",
        "a sketch of a {class_name}",
        "a painting of a {class_name}",
        "a clipart image of a {class_name}",
        "a domain-invariant representation of a {class_name}",
        "a {class_name} independent of background, texture and style",
        "a {class_name} with complete object structure and recognizable shape",
    ],
    # CLIP recommended ImageNet-style prompt ensemble used by RISE.
    # These templates intentionally use `{class}` as the placeholder.
    "rise80": [
        'a bad photo of a {class}.',
        'a photo of many {class}.',
        'a sculpture of a {class}.',
        'a photo of the hard to see {class}.',
        'a low resolution photo of the {class}.',
        'a rendering of a {class}.',
        'graffiti of a {class}.',
        'a bad photo of the {class}.',
        'a cropped photo of the {class}.',
        'a tattoo of a {class}.',
        'the embroidered {class}.',
        'a photo of a hard to see {class}.',
        'a bright photo of a {class}.',
        'a photo of a clean {class}.',
        'a photo of a dirty {class}.',
        'a dark photo of the {class}.',
        'a drawing of a {class}.',
        'a photo of my {class}.',
        'the plastic {class}.',
        'a photo of the cool {class}.',
        'a close-up photo of a {class}.',
        'a black and white photo of the {class}.',
        'a painting of the {class}.',
        'a painting of a {class}.',
        'a pixelated photo of the {class}.',
        'a sculpture of the {class}.',
        'a bright photo of the {class}.',
        'a cropped photo of a {class}.',
        'a plastic {class}.',
        'a photo of the dirty {class}.',
        'a jpeg corrupted photo of a {class}.',
        'a blurry photo of the {class}.',
        'a photo of the {class}.',
        'a good photo of the {class}.',
        'a rendering of the {class}.',
        'a {class} in a video game.',
        'a photo of one {class}.',
        'a doodle of a {class}.',
        'a close-up photo of the {class}.',
        'a photo of a {class}.',
        'the origami {class}.',
        'the {class} in a video game.',
        'a sketch of a {class}.',
        'a doodle of the {class}.',
        'a origami {class}.',
        'a low resolution photo of a {class}.',
        'the toy {class}.',
        'a rendition of the {class}.',
        'a photo of the clean {class}.',
        'a photo of a large {class}.',
        'a rendition of a {class}.',
        'a photo of a nice {class}.',
        'a photo of a weird {class}.',
        'a blurry photo of a {class}.',
        'a cartoon {class}.',
        'art of a {class}.',
        'a sketch of the {class}.',
        'a embroidered {class}.',
        'a pixelated photo of a {class}.',
        'itap of the {class}.',
        'a jpeg corrupted photo of the {class}.',
        'a good photo of a {class}.',
        'a plushie {class}.',
        'a photo of the nice {class}.',
        'a photo of the small {class}.',
        'a photo of the weird {class}.',
        'the cartoon {class}.',
        'art of the {class}.',
        'a drawing of the {class}.',
        'a photo of the large {class}.',
        'a black and white photo of a {class}.',
        'the plushie {class}.',
        'a dark photo of a {class}.',
        'itap of a {class}.',
        'graffiti of the {class}.',
        'a toy {class}.',
        'itap of my {class}.',
        'a photo of a cool {class}.',
        'a photo of a small {class}.',
        'a tattoo of the {class}.',
    ],
}


def _import_clip():
    try:
        import clip  # type: ignore
        return clip
    except ImportError as exc:
        repo_root = Path(__file__).resolve().parents[3]
        local_clip = repo_root / "CLIP"
        if local_clip.exists():
            sys.path.insert(0, str(local_clip))
            try:
                import clip  # type: ignore
                return clip
            except ImportError:
                pass
        raise ImportError(
            "RISE-guided DCCL requires OpenAI CLIP. Install it with "
            "`pip install git+https://github.com/openai/CLIP.git` or keep the "
            "repo-local CLIP/ package importable."
        ) from exc


def normalize_class_name(class_name):
    return class_name.replace("_", " ").replace("-", " ").strip()


def prompt_count(prompt_mode):
    if prompt_mode not in PROMPT_TEMPLATES:
        raise ValueError(
            f"Unsupported --rise_prompt_mode={prompt_mode!r}. "
            f"Choose one of {sorted(PROMPT_TEMPLATES)}."
        )
    return len(PROMPT_TEMPLATES[prompt_mode])


def build_prompts(class_names, prompt_mode):
    prompt_count(prompt_mode)
    prompts = []
    templates = PROMPT_TEMPLATES[prompt_mode]
    for class_name in class_names:
        clean_name = normalize_class_name(class_name)
        if not clean_name:
            raise ValueError(f"Empty class name after normalization: {class_name!r}")
        prompts.append([
            template.format_map({"class": clean_name, "class_name": clean_name})
            for template in templates
        ])
    return prompts


def load_clip_teacher(model_name, device="cpu", freeze=True, download_root=None):
    clip = _import_clip()
    try:
        clip_model, _preprocess = clip.load(
            model_name, device=device, jit=False, download_root=download_root
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load CLIP model {model_name!r} for RISE-guided DCCL. "
            "If this server has no internet access, pre-place the CLIP .pt "
            "file under --rise_clip_download_root (for model names such as "
            "ViT-B/32) or pass the checkpoint file path directly via "
            "--rise_clip_model_name."
        ) from exc

    clip_model.eval()
    if freeze:
        for parameter in clip_model.parameters():
            parameter.requires_grad = False
    return clip_model, clip


@torch.no_grad()
def build_text_prototypes(clip_model, clip_module, class_names, prompt_mode, device):
    prompts_by_class = build_prompts(class_names, prompt_mode)
    prototypes = []
    for prompts in prompts_by_class:
        tokens = clip_module.tokenize(prompts).to(device)
        text_features = clip_model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=-1)
        prototype = F.normalize(text_features.mean(dim=0), dim=0)
        prototypes.append(prototype)
    if not prototypes:
        raise ValueError("Cannot build RISE text prototypes because no class names were provided.")
    return torch.stack(prototypes, dim=0)


class CLIPInputNormalize(nn.Module):
    """Convert ImageNet-normalized DCCL tensors to CLIP-normalized tensors."""

    def __init__(self):
        super().__init__()
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = x * self.imagenet_std + self.imagenet_mean
        x = x.clamp(0.0, 1.0)
        return (x - self.clip_mean) / self.clip_std


def clip_kd_loss(student_logits, teacher_logits, temperature):
    teacher_logits = teacher_logits.detach()
    log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_probs, teacher_probs, reduction="batchmean") * temperature * temperature


def prototype_alignment_loss(projected_features, labels, text_prototypes):
    z_s = F.normalize(projected_features, dim=1)
    proto_y = text_prototypes.detach()[labels]
    cosine = F.cosine_similarity(z_s, proto_y, dim=1)
    return (1.0 - cosine).mean(), cosine.mean()
