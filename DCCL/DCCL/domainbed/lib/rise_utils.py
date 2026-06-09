"""Utilities for RISE-guided CLIP semantic supervision.

This module keeps CLIP loading, prompt construction, CLIP preprocessing, and
RISE loss computations outside DCCL.update() so the default DCCL/ICR path stays
unchanged when --use_rise is disabled.
"""

from pathlib import Path
import sys
import importlib.util

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def _find_repo_clip_path():
    """Return the repository-local CLIP package path if present."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "CLIP"
        if (candidate / "clip" / "clip.py").exists():
            return candidate
    return None


def load_clip_teacher(model_name="ViT-B/32", download_root=None, freeze=True, device="cpu"):
    """Load a frozen OpenAI CLIP teacher with a clear offline/error message."""
    repo_clip = _find_repo_clip_path()
    if repo_clip is not None and str(repo_clip) not in sys.path:
        sys.path.insert(0, str(repo_clip))
    if importlib.util.find_spec("clip") is None:
        raise RuntimeError(
            "RISE requires the OpenAI CLIP package. Install CLIP or keep the "
            "repository-local CLIP package available before running with --use_rise."
        )
    import clip  # noqa: WPS433 - optional dependency loaded only for RISE

    try:
        clip_model, _ = clip.load(model_name, device=device, download_root=download_root)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load CLIP model '{model_name}' with download_root={download_root!r}. "
            "If the server has no internet access, provide a local CLIP cache or a "
            "checkpoint path via --rise_clip_download_root."
        ) from exc

    if not _as_bool(freeze):
        raise ValueError("Stage-1 RISE requires --rise_freeze_clip true so CLIP stays frozen.")
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval()
    return clip_model, clip


def get_rise80_templates():
    """OpenAI CLIP ImageNet-style 80 prompt templates."""
    return [
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
    ]


def get_prompt_templates(prompt_mode):
    if prompt_mode == "simple":
        return ["a photo of a {}"]
    if prompt_mode == "multi":
        return [
            "a photo of a {}",
            "a sketch of a {}",
            "a painting of a {}",
            "a clipart image of a {}",
            "a drawing of a {}",
        ]
    if prompt_mode == "domain_invariant":
        return [
            "a photo of a {}",
            "a sketch of a {}",
            "a painting of a {}",
            "a clipart image of a {}",
            "a domain-invariant representation of a {}",
            "a {} independent of background, texture and style",
            "a {} with complete object structure and recognizable shape",
        ]
    if prompt_mode == "rise80":
        return get_rise80_templates()
    raise ValueError("rise_prompt_mode must be one of: simple, multi, domain_invariant, rise80.")


def clean_class_names(class_names, expected_num_classes=None):
    if class_names is None:
        raise ValueError(
            "RISE requires class names aligned with label indices. Provide dataset.classes "
            "or a class mapping file before running with --use_rise."
        )
    names = [str(name).replace("_", " ").strip() for name in class_names]
    if expected_num_classes is not None and len(names) != expected_num_classes:
        raise ValueError(
            f"RISE class-name count mismatch: got {len(names)} names for "
            f"{expected_num_classes} classes. Ensure DomainNet class mapping or "
            "dataset.classes is aligned with label indices."
        )
    return names


@torch.no_grad()
def build_text_prototypes(clip_model, clip_module, class_names, prompt_mode="rise80", device="cpu"):
    """Encode prompt ensembles and return detached [num_classes, clip_dim] prototypes."""
    templates = get_prompt_templates(prompt_mode)
    prototypes = []
    clip_model.eval()
    for class_name in class_names:
        prompts = [template.format(class_name) for template in templates]
        tokens = clip_module.tokenize(prompts).to(device)
        text_features = clip_model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=-1)
        prototype = F.normalize(text_features.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        prototypes.append(prototype)
    return torch.stack(prototypes, dim=0).detach()


def preprocess_for_clip(x):
    """Convert ImageNet-normalized DCCL inputs into CLIP-normalized tensors."""
    device = x.device
    dtype = x.dtype
    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, -1, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, -1, 1, 1)
    clip_mean = torch.tensor(CLIP_MEAN, device=device, dtype=dtype).view(1, -1, 1, 1)
    clip_std = torch.tensor(CLIP_STD, device=device, dtype=dtype).view(1, -1, 1, 1)
    x_01 = (x * imagenet_std + imagenet_mean).clamp(0.0, 1.0)
    return (x_01 - clip_mean) / clip_std


@torch.no_grad()
def compute_clip_logits(clip_model, x, text_prototypes):
    clip_model.eval()
    x_clip = preprocess_for_clip(x)
    clip_img_feat = clip_model.encode_image(x_clip).float()
    clip_img_feat = F.normalize(clip_img_feat, dim=-1)
    logit_scale = clip_model.logit_scale.exp().float()
    return (logit_scale * clip_img_feat @ text_prototypes.float().t()).detach()


def clip_kd_loss(student_logits, clip_logits, temperature):
    if student_logits.shape[1] != clip_logits.shape[1]:
        raise ValueError(
            f"RISE KD class dimension mismatch: student logits have {student_logits.shape[1]} "
            f"classes but CLIP teacher logits have {clip_logits.shape[1]} classes."
        )
    temp = max(float(temperature), 1e-6)
    return F.kl_div(
        F.log_softmax(student_logits / temp, dim=1),
        F.softmax(clip_logits.detach() / temp, dim=1),
        reduction="batchmean",
    ) * temp * temp


def proto_alignment_loss(student_features, labels, text_prototypes, projector):
    z_s = F.normalize(projector(student_features), dim=-1)
    proto_y = text_prototypes.detach()[labels]
    if z_s.shape[1] != proto_y.shape[1]:
        raise ValueError(
            f"RISE prototype dimension mismatch: projected student dim={z_s.shape[1]} "
            f"but text prototype dim={proto_y.shape[1]}. Set --rise_projection_dim accordingly."
        )
    cosine = F.cosine_similarity(z_s, proto_y.detach(), dim=-1)
    return (1.0 - cosine).mean(), cosine.mean()
