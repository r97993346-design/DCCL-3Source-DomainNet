"""RISE external CLIP teacher for DCCL.

RISE is train-only guidance: frozen CLIP image/text encoders build teacher
logits and text prototypes.  Evaluation still uses only the DCCL student.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


RISE80_PROMPTS = [
    "a photo of a {class}.", "a bad photo of a {class}.", "a photo of many {class}.",
    "a sculpture of a {class}.", "a photo of the hard to see {class}.", "a low resolution photo of a {class}.",
    "a rendering of a {class}.", "graffiti of a {class}.", "a bad photo of the {class}.",
    "a cropped photo of the {class}.", "a tattoo of a {class}.", "the embroidered {class}.",
    "a photo of a hard to see {class}.", "a bright photo of a {class}.", "a photo of a clean {class}.",
    "a photo of a dirty {class}.", "a dark photo of the {class}.", "a drawing of a {class}.",
    "a photo of my {class}.", "the plastic {class}.", "a photo of the cool {class}.",
    "a close-up photo of a {class}.", "a black and white photo of the {class}.",
    "a painting of the {class}.", "a painting of a {class}.", "a pixelated photo of the {class}.",
    "a sculpture of the {class}.", "a bright photo of the {class}.", "a cropped photo of a {class}.",
    "a plastic {class}.", "a photo of the dirty {class}.", "a jpeg corrupted photo of a {class}.",
    "a blurry photo of the {class}.", "a photo of the {class}.", "a good photo of the {class}.",
    "a rendering of the {class}.", "a {class} in a video game.", "a photo of one {class}.",
    "a doodle of a {class}.", "a close-up photo of the {class}.", "a photo of a {class} in a scene.",
    "a photo of the clean {class}.", "a photo of a large {class}.", "a photo of a nice {class}.",
    "a photo of a weird {class}.", "a blurry photo of a {class}.", "a cartoon {class}.",
    "art of a {class}.", "a sketch of the {class}.", "a embroidered {class}.",
    "a pixelated photo of a {class}.", "itap of the {class}.", "a jpeg corrupted photo of the {class}.",
    "a good photo of a {class}.", "a plushie {class}.", "a photo of the nice {class}.",
    "a photo of the small {class}.", "a photo of the weird {class}.", "the cartoon {class}.",
    "art of the {class}.", "a drawing of the {class}.", "a photo of the large {class}.",
    "a black and white photo of a {class}.", "the plushie {class}.", "a dark photo of a {class}.",
    "itap of a {class}.", "graffiti of the {class}.", "a toy {class}.", "itap of my {class}.",
    "a photo of a cool {class}.", "a photo of a small {class}.", "a tattoo of the {class}.",
    "a sketch of a {class}.", "a doodle of the {class}.", "a photo of a domain-invariant {class}.",
    "a clipart image of a {class}.", "a painting style image of a {class}.",
    "a sketch style image of a {class}.", "a real-world photo of a {class}.",
    "a semantic representation of a {class}.",
]


class RISETeacher(nn.Module):
    def __init__(self, class_names, student_dim, clip_model_name="ViT-B/32", download_root=None, prompt_mode="rise80"):
        super().__init__()
        self.prompt_mode = prompt_mode
        self.clip_model_name = clip_model_name
        self.download_root = download_root
        clip_root = Path(__file__).resolve().parents[4] / "CLIP"
        if str(clip_root) not in sys.path:
            sys.path.insert(0, str(clip_root))
        try:
            import clip
        except ImportError as exc:
            raise RuntimeError("RISE requires the local OpenAI CLIP package. Ensure ./CLIP is present or install clip.") from exc

        clip_models = getattr(__import__("clip.clip", fromlist=["_MODELS"]), "_MODELS", {})
        if clip_model_name in clip_models and not os.path.isfile(clip_model_name):
            cache_root = download_root or os.path.expanduser("~/.cache/clip")
            url = clip_models[clip_model_name]
            cached = os.path.join(cache_root, os.path.basename(url))
            if not os.path.isfile(cached):
                raise RuntimeError(
                    f"CLIP checkpoint for '{clip_model_name}' was not found at {cached}. "
                    f"RISE does not force network downloads; set --rise_clip_download_root to a "
                    f"directory containing the cached CLIP weights, or pass --rise_clip_model_name "
                    f"as a local checkpoint path."
                )
        try:
            self.clip_model, _ = clip.load(clip_model_name, device="cpu", jit=False, download_root=download_root)
        except Exception as exc:
            hint = (
                f"Failed to load CLIP model '{clip_model_name}'. Place the CLIP checkpoint in "
                f"--rise_clip_download_root (current: {download_root}) or pass "
                f"--rise_clip_model_name as a local checkpoint path."
            )
            raise RuntimeError(hint) from exc

        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip = clip
        self.text_prototypes = None
        self.clip_dim = self.clip_model.text_projection.shape[1]
        self.align_proj = nn.Linear(student_dim, self.clip_dim)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.build_text_prototypes(class_names)

    def templates(self):
        if self.prompt_mode == "rise80":
            return RISE80_PROMPTS
        if self.prompt_mode == "single":
            return ["a photo of a {class}."]
        raise ValueError(f"Unknown --rise_prompt_mode={self.prompt_mode}")

    @torch.no_grad()
    def build_text_prototypes(self, class_names):
        device = next(self.clip_model.parameters()).device
        prototypes = []
        for class_name in class_names:
            clean_name = str(class_name).replace("_", " ")
            prompts = [template.format(**{"class": clean_name}) for template in self.templates()]
            tokens = self.clip.tokenize(prompts, truncate=True).to(device)
            text_features = self.clip_model.encode_text(tokens).float()
            text_features = F.normalize(text_features, dim=-1)
            prototypes.append(F.normalize(text_features.mean(dim=0), dim=0))
        self.text_prototypes = F.normalize(torch.stack(prototypes, dim=0), dim=-1)

    def _clip_normalize(self, images):
        images = images * self.imagenet_std.to(images.device) + self.imagenet_mean.to(images.device)
        images = images.clamp(0.0, 1.0)
        return (images - self.clip_mean.to(images.device)) / self.clip_std.to(images.device)

    @torch.no_grad()
    def compute_teacher_logits(self, images, temperature=1.0):
        self.clip_model.eval()
        images = self._clip_normalize(images).type(next(self.clip_model.parameters()).dtype)
        image_features = self.clip_model.encode_image(images).float()
        image_features = F.normalize(image_features, dim=-1)
        prototypes = self.text_prototypes.to(image_features.device)
        return image_features @ prototypes.T / max(float(temperature), 1e-6)

    def compute_kd_loss(self, student_logits, teacher_logits, tau=2.0):
        tau = max(float(tau), 1e-6)
        return F.kl_div(
            F.log_softmax(student_logits / tau, dim=1),
            F.softmax(teacher_logits / tau, dim=1),
            reduction="batchmean",
        ) * (tau ** 2)

    def compute_ad_loss(self, student_z, labels):
        z_align = F.normalize(self.align_proj(student_z), dim=-1)
        target_proto = self.text_prototypes.to(z_align.device)[labels]
        return 1.0 - F.cosine_similarity(z_align, target_proto, dim=-1).mean()
