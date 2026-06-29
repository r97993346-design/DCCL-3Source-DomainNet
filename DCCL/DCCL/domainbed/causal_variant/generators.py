import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F

from .prompt_bank import build_diffusion_edit_prompt, get_style_prompt_bank

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _as_bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def _slug(value):
    value = str(value) if value is not None else "unknown"
    value = value.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value) or "unknown"


def _norm_tensors(device, dtype):
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def denormalize(x):
    mean, std = _norm_tensors(x.device, x.dtype)
    y = x * std + mean
    if y.min().item() < -0.05 or y.max().item() > 1.05:
        return x.clamp(0, 1)
    return y.clamp(0, 1)


def normalize_like(x01, ref):
    if ref.min().detach().item() >= -0.05 and ref.max().detach().item() <= 1.05:
        return x01.to(dtype=ref.dtype, device=ref.device)
    mean, std = _norm_tensors(ref.device, ref.dtype)
    return (x01.to(dtype=ref.dtype, device=ref.device) - mean) / std


def _pil_to_tensor(image, device):
    if image.mode != "RGB":
        image = image.convert("RGB")
    raw = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    return raw.view(image.size[1], image.size[0], 3).permute(2, 0, 1).float().div(255).to(device)


def _tensor_to_pil(image):
    array = denormalize(image.unsqueeze(0)).squeeze(0).detach().cpu().mul(255).clamp(0, 255).byte()
    array = array.permute(1, 2, 0).numpy()
    return Image.fromarray(array)


@dataclass
class VariantBatch:
    images: torch.Tensor
    source_indices: torch.Tensor
    kinds: list
    metadata: list = field(default_factory=list)


class PhotometricVariantGenerator:
    def __init__(self, ops, strength_min=0.1, strength_max=0.5):
        self.ops = [op.strip() for op in str(ops).split(',') if op.strip()]
        self.strength_min = float(strength_min)
        self.strength_max = float(strength_max)

    @torch.no_grad()
    def __call__(self, x):
        x01 = denormalize(x)
        out = []
        for img in x01:
            op = random.choice(self.ops) if self.ops else "brightness"
            s = random.uniform(self.strength_min, self.strength_max)
            y = img.clone()
            if op == "brightness":
                y = (y * (1.0 + random.choice([-1, 1]) * s)).clamp(0, 1)
            elif op == "contrast":
                mean = y.mean(dim=(1, 2), keepdim=True)
                y = ((y - mean) * (1.0 + random.choice([-1, 1]) * s) + mean).clamp(0, 1)
            elif op in ("color", "color_jitter"):
                factors = torch.empty(3, 1, 1, device=y.device, dtype=y.dtype).uniform_(1-s, 1+s)
                y = (y * factors).clamp(0, 1)
            elif op == "sharpness":
                blur = F.avg_pool2d(y.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
                y = (y + s * (y - blur)).clamp(0, 1)
            elif op == "gaussian_noise":
                y = (y + torch.randn_like(y) * s * 0.25).clamp(0, 1)
            elif op == "blur":
                y = F.avg_pool2d(y.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
            out.append(y)
        return normalize_like(torch.stack(out, 0), x)


class XDomainMixVariantGenerator:
    def __init__(self, alpha=0.5, same_class_only=True, require_diff_domain=True, fallback_skip=True):
        self.alpha = float(alpha)
        self.same_class_only = _as_bool(same_class_only)
        self.require_diff_domain = _as_bool(require_diff_domain)
        self.fallback_skip = _as_bool(fallback_skip)

    @torch.no_grad()
    def __call__(self, x, y, domains=None):
        x01 = denormalize(x)
        imgs, idxs = [], []
        n = x.shape[0]
        for i in range(n):
            candidates = torch.arange(n, device=x.device)
            mask = candidates != i
            if self.same_class_only:
                mask &= y == y[i]
            if domains is not None and self.require_diff_domain:
                mask &= domains != domains[i]
            donor_idxs = candidates[mask]
            if donor_idxs.numel() == 0:
                if domains is not None and self.require_diff_domain and not self.fallback_skip:
                    mask = (candidates != i) & ((y == y[i]) if self.same_class_only else True)
                    donor_idxs = candidates[mask]
                if donor_idxs.numel() == 0:
                    continue
            j = donor_idxs[torch.randint(0, donor_idxs.numel(), (1,), device=x.device)].item()
            src, donor = x01[i:i+1], x01[j:j+1]
            sm, ss = src.mean((2, 3), keepdim=True), src.std((2, 3), keepdim=True).clamp_min(1e-6)
            dm, ds = donor.mean((2, 3), keepdim=True), donor.std((2, 3), keepdim=True).clamp_min(1e-6)
            styled = (src - sm) / ss * ds + dm
            mixed = ((1 - self.alpha) * src + self.alpha * styled).clamp(0, 1).squeeze(0)
            imgs.append(mixed); idxs.append(i)
        if not imgs:
            return None
        return VariantBatch(normalize_like(torch.stack(imgs, 0), x), torch.tensor(idxs, device=x.device), ["xdomainmix"] * len(imgs), [{} for _ in imgs])


class DiffuseMixVariantGenerator:
    def __init__(self, hparams):
        self.hparams = hparams
        self.pipe = None
        self.styles = get_style_prompt_bank(hparams.get("causal_prompt_bank", "default_dg_style"))
        if hparams.get("causal_prompt_mode", "target_agnostic") == "target_aware":
            print("WARNING: target-aware prompt mode uses target domain prior and is not strict DG.")
        model_path = hparams.get("causal_diffusion_model_path", "")
        if not model_path:
            raise ValueError("causal_use_diffusion=True requires --causal_diffusion_model_path for online DiffuseMix generation")
        try:
            from diffusers import StableDiffusionInstructPix2PixPipeline
            self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_path, local_files_only=_as_bool(hparams.get("causal_diffusion_local_only", True))
            ).to(hparams.get("causal_diffusion_device", "cuda"))
            for component_name in ("unet", "vae", "text_encoder", "safety_checker"):
                component = getattr(self.pipe, component_name, None)
                if component is None:
                    continue
                if hasattr(component, "eval"):
                    component.eval()
                if hasattr(component, "parameters"):
                    for param in component.parameters():
                        param.requires_grad_(False)
            if hasattr(self.pipe, "set_progress_bar_config"):
                self.pipe.set_progress_bar_config(disable=True)
        except Exception as exc:
            raise RuntimeError("Failed to initialize online DiffuseMix pipeline: {}".format(exc))

    def _record_for(self, source_idx, label, class_name, domain_name, original_path, image_size, step):
        style = random.choice(self.styles)
        prompt = build_diffusion_edit_prompt(class_name, style)
        seed = int(self.hparams.get("causal_diffusion_seed", 0)) + int(step)
        original_key = original_path or "sample-{}".format(source_idx)
        prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
        size_tag = "{}x{}".format(image_size[-1], image_size[-2])
        cache_key = hashlib.sha1("|".join([original_key, style, prompt_hash, str(seed), size_tag]).encode("utf-8")).hexdigest()[:16]
        dataset = _slug(self.hparams.get("dataset", "dataset"))
        domain = _slug(domain_name)
        cls = _slug(class_name)
        stem = _slug(Path(original_key).stem)
        style_tag = _slug(style)[:48]
        filename = "{}__style-{}__seed{}__step{}__{}.jpg".format(stem, style_tag, seed, step, cache_key)
        mode = self.hparams.get("causal_save_diffusion_mode", "kept")
        save_mode_dir = "kept" if mode in ("kept", "none") else mode
        cache_dir = Path(self.hparams.get("causal_diffusion_cache_dir", "causal_diffusion_cache"))
        cf_path = cache_dir / dataset / domain / cls / save_mode_dir / filename
        return {
            "source_index": int(source_idx), "original_path": original_path or original_key, "label": int(label),
            "class_name": class_name, "domain": domain_name, "style": style, "prompt": prompt,
            "seed": seed, "step": int(step), "cache_key": cache_key, "cf_path": str(cf_path),
        }

    @torch.no_grad()
    def __call__(self, x, y, class_names=None, step=0, original_paths=None, domain_names=None):
        every = int(self.hparams.get("causal_diffusion_every_n_steps", 1))
        if every > 1 and step % every != 0:
            return None
        max_n = min(int(self.hparams.get("causal_diffusion_max_images_per_step", 4)), x.shape[0])
        if max_n <= 0:
            return None
        idxs = torch.arange(max_n, device=x.device)
        x01 = denormalize(x[idxs])
        try:
            from torchvision.transforms.functional import to_pil_image
            pil, prompts, records, cached_tensors, generate_positions = [], [], [], [], []
            labels = y[idxs].detach().cpu().tolist()
            for pos, label in enumerate(labels):
                src_idx = int(idxs[pos].item())
                cls = class_names[src_idx] if class_names and src_idx < len(class_names) else str(label)
                domain = domain_names[src_idx] if domain_names and src_idx < len(domain_names) else "domain_{}".format(src_idx)
                original_path = original_paths[src_idx] if original_paths and src_idx < len(original_paths) else None
                rec = self._record_for(src_idx, label, cls, domain, original_path, x.shape[-2:], step)
                records.append(rec)
                cached = None
                if _as_bool(self.hparams.get("causal_use_diffusion_cache", True)):
                    cached_path = Path(rec["cf_path"])
                    if cached_path.exists():
                        try:
                            cached = _pil_to_tensor(Image.open(cached_path), x.device)
                        except Exception as exc:
                            print("WARNING: failed to read cached diffusion image {}; regenerating: {}".format(cached_path, exc))
                if cached is None:
                    pil.append(to_pil_image(x01[pos].cpu()))
                    prompts.append(rec["prompt"])
                    generate_positions.append(pos)
                    cached_tensors.append(None)
                else:
                    cached_tensors.append(cached)
            if generate_positions:
                device = self.hparams.get("causal_diffusion_device", "cuda")
                gen = torch.Generator(device=device).manual_seed(int(self.hparams.get("causal_diffusion_seed", 0)) + int(step))
                out = self.pipe(
                    prompt=prompts, image=pil, num_inference_steps=int(self.hparams.get("causal_diffusion_steps", 50)),
                    guidance_scale=float(self.hparams.get("causal_diffusion_cfg_text", 8.5)),
                    image_guidance_scale=float(self.hparams.get("causal_diffusion_cfg_image", 1.1)), generator=gen,
                ).images
                for pos, image in zip(generate_positions, out):
                    cached_tensors[pos] = _pil_to_tensor(image, x.device)
            imgs = torch.stack(cached_tensors, 0)
            return VariantBatch(normalize_like(imgs, x), idxs, ["diffusion"] * len(records), records)
        except Exception as exc:
            print("WARNING: online DiffuseMix generation failed; skipping diffusion candidates: {}".format(exc))
            return None


class CausalVariantGenerator:
    def __init__(self, hparams):
        self.hparams = hparams
        self.photo = PhotometricVariantGenerator(hparams.get("causal_photo_ops"), hparams.get("causal_photo_strength_min"), hparams.get("causal_photo_strength_max")) if _as_bool(hparams.get("causal_use_photometric", True)) else None
        self.xdm = XDomainMixVariantGenerator(hparams.get("causal_xdomainmix_alpha"), hparams.get("causal_xdomainmix_same_class_only"), hparams.get("causal_xdomainmix_require_diff_domain"), hparams.get("causal_xdomainmix_fallback_skip")) if _as_bool(hparams.get("causal_use_xdomainmix", True)) else None
        self.diff = DiffuseMixVariantGenerator(hparams) if _as_bool(hparams.get("causal_use_diffusion", True)) else None

    @torch.no_grad()
    def __call__(self, x, y, domains=None, step=0, class_names=None, original_paths=None, domain_names=None):
        batches = []
        if self.photo is not None:
            imgs = self.photo(x)
            batches.append(VariantBatch(imgs, torch.arange(x.shape[0], device=x.device), ["photo"] * x.shape[0], [{} for _ in range(x.shape[0])]))
        if self.xdm is not None:
            b = self.xdm(x, y, domains)
            if b is not None: batches.append(b)
        if self.diff is not None:
            b = self.diff(x, y, class_names, step, original_paths, domain_names)
            if b is not None: batches.append(b)
        if not batches:
            return None
        return VariantBatch(
            torch.cat([b.images for b in batches], 0),
            torch.cat([b.source_indices for b in batches], 0),
            sum([b.kinds for b in batches], []),
            sum([b.metadata for b in batches], []),
        )


def save_diffusion_images(variant_batch, save_indices, anchor_sim=None, cls_conf=None, kept_anchor=None,
                          kept_cls=None, kept_after_filter=None, selected_indices=None,
                          save_metadata=True):
    selected_set = set(int(i) for i in (selected_indices.detach().cpu().tolist() if torch.is_tensor(selected_indices) else (selected_indices or [])))
    indices = save_indices.detach().cpu().tolist() if torch.is_tensor(save_indices) else list(save_indices)
    for idx in indices:
        if idx < 0 or idx >= len(variant_batch.metadata) or variant_batch.kinds[idx] != "diffusion":
            continue
        rec = dict(variant_batch.metadata[idx])
        cf_path = Path(rec.get("cf_path", ""))
        if not cf_path:
            continue
        rec.update({
            "cf_path": str(cf_path),
            "anchor_sim": None if anchor_sim is None else float(anchor_sim[idx].detach().cpu().item()),
            "cls_conf": None if cls_conf is None else float(cls_conf[idx].detach().cpu().item()),
            "kept_by_anchor": True if kept_anchor is None else bool(kept_anchor[idx].detach().cpu().item()),
            "kept_by_cls": True if kept_cls is None else bool(kept_cls[idx].detach().cpu().item()),
            "kept_after_filter": True if kept_after_filter is None else bool(kept_after_filter[idx].detach().cpu().item()),
            "selected_as_hard_positive": int(idx) in selected_set,
        })
        try:
            cf_path.parent.mkdir(parents=True, exist_ok=True)
            _tensor_to_pil(variant_batch.images[idx]).save(cf_path, quality=95)
        except Exception as exc:
            print("WARNING: failed to save diffusion image {}: {}".format(cf_path, exc))
            continue
        if save_metadata:
            try:
                with open(cf_path.parent / "metadata.jsonl", "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as exc:
                print("WARNING: failed to write diffusion metadata for {}: {}".format(cf_path, exc))
