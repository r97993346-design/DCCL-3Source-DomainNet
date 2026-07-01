import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

DIFFUSEMIX_OFFICIAL_PROMPTS = [
    "Autumn", "snowy", "watercolor art", "sunset", "rainbow",
    "aurora", "mosaic", "ukiyo-e", "a sketch with crayon",
]


_basic = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_to_pil = T.ToPILImage()
_to_tensor = T.ToTensor()
_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denorm_to_pil(x):
    x = x.detach().cpu().unsqueeze(0)
    img = (x * _std + _mean).clamp(0, 1).squeeze(0)
    return _to_pil(img)


def safe_name(s):
    return str(s).replace("/", "_").replace(" ", "_")


def image_id_from_path(path, fallback):
    if path:
        return Path(path).stem
    return f"idx_{int(fallback):06d}"


class DiffuseMixPositiveManager:
    """Cache/generate/filter DiffuseMix causal positives with conservative defaults."""
    def __init__(self, args, dataset_name, class_names, device, logger=None):
        self.args = args
        self.dataset_name = dataset_name
        self.class_names = class_names or []
        self.device = device
        self.logger = logger
        self.pipe = None
        self.fractal_images = None
        self.basic = _basic
        self.stats = defaultdict(float)
        self.env_attempts = defaultdict(int)
        self.env_kept = defaultdict(int)
        self.env_strong = defaultdict(int)
        self.class_attempts = defaultdict(int)
        self.class_kept = defaultdict(int)
        self.class_strong = defaultdict(int)

    def _prompt_list(self):
        raw = getattr(self.args, "diffusemix_prompts", "")
        prompts = [p.strip() for p in raw.split(",") if p.strip()]
        return prompts if prompts else DIFFUSEMIX_OFFICIAL_PROMPTS

    def prompt(self, seed=None):
        prompts = self._prompt_list()
        if seed is None:
            return prompts[0]
        return prompts[int(seed) % len(prompts)]

    def _cache_dir(self, source_env, class_name, image_id):
        return (Path(self.args.diffusemix_cache_dir) / self.dataset_name /
                f"source_env_{int(source_env)}" / safe_name(class_name) / safe_name(image_id))

    def _class_name(self, y):
        return self.class_names[int(y)] if int(y) < len(self.class_names) else str(int(y))

    def _load_cached(self, source_env, class_name, image_id):
        d = self._cache_dir(source_env, class_name, image_id)
        if not d.exists():
            return []
        items = []
        for meta_path in sorted(d.glob("sample_*.json")):
            try:
                meta = json.loads(meta_path.read_text())
                if not meta.get("filter_pass", False):
                    continue
                cached_mode = meta.get("augmentation_mode", "direct")
                current_mode = getattr(self.args, "diffusemix_augmentation_mode", "diffusemix")
                if cached_mode != current_mode:
                    continue
                if current_mode == "diffusemix" and not meta.get("diffusemix_source_style", False):
                    continue
                if current_mode == "diffusemix" and meta.get("diffusemix_composition_version") != "paper_quality_v3":
                    continue
                img_path = meta_path.with_suffix(".png")
                if not img_path.exists():
                    continue
                items.append((img_path, meta))
            except Exception:
                continue
        items.sort(key=lambda p: (not p[1].get("strong_positive", False), str(p[0])))
        return items[: max(1, int(self.args.diffusemix_max_cached_used_per_image))]

    def _ensure_pipe(self):
        if self.pipe is not None:
            return self.pipe
        if not self.args.diffusemix_model_path:
            raise RuntimeError("--diffusemix_model_path is required when cache is empty and regeneration is enabled")
        from diffusers import StableDiffusionInstructPix2PixPipeline
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.args.diffusemix_model_path, torch_dtype=torch.float16 if str(self.device).startswith("cuda") else torch.float32
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        # Diffusers pipelines are containers, not always nn.Modules. Some
        # versions expose ``parameters`` as a dict-like property instead of a
        # callable method, so freeze trainable submodules through components.
        for component in self.pipe.components.values():
            if hasattr(component, "eval"):
                component.eval()
            if hasattr(component, "parameters"):
                for param in component.parameters():
                    param.requires_grad_(False)
        return self.pipe

    def _load_fractal_images(self):
        if self.fractal_images is not None:
            return self.fractal_images
        fractal_dir = getattr(self.args, "diffusemix_fractal_dir", "")
        if not getattr(self.args, "diffusemix_use_real_fractal", True):
            raise RuntimeError("Source-style DiffuseMix requires --diffusemix_use_real_fractal and real images from --diffusemix_fractal_dir")
        if not fractal_dir:
            raise RuntimeError("--diffusemix_fractal_dir is required when --diffusemix_augmentation_mode diffusemix is used for new generation")
        exts = (".png", ".jpg", ".jpeg")
        paths = sorted([Path(fractal_dir) / name for name in os.listdir(fractal_dir) if name.lower().endswith(exts)])
        if not paths:
            raise RuntimeError(f"No fractal images found in --diffusemix_fractal_dir={fractal_dir}")
        self.fractal_images = [Image.open(path).convert("RGB").resize((256, 256)) for path in paths]
        return self.fractal_images

    def _source_style_combine_images(self, original_img, generated_img, seed):
        """Build the DIFFUSEMIX hybrid image H = M*I + (1-M)*I_hat."""
        original_img = original_img.convert("RGB").resize((256, 256))
        generated_img = generated_img.convert("RGB").resize((256, 256))
        width, height = original_img.size
        vertical = True
        split = width // 2
        keep_original_on_first_side = True

        # Use a hard left/right mask so the result is visibly a splice: the
        # left half comes from the source image, the right half from the
        # diffusion-generated image. This matches the paper's illustrated
        # source-style DIFFUSEMIX pipeline and avoids the previous transparent
        # overlay / top-bottom artifact.
        mask = np.zeros((height, width), dtype=np.float32)
        mask[:, :split] = 1.0
        mask = mask[:, :, np.newaxis]
        original_array = np.array(original_img, dtype=np.float32)
        generated_array = np.array(generated_img, dtype=np.float32)
        hybrid_array = mask * original_array + (1.0 - mask) * generated_array
        return Image.fromarray(np.clip(hybrid_array, 0, 255).astype(np.uint8)), vertical, split, keep_original_on_first_side

    def _source_style_blend_fractal(self, base_img, fractal_img, alpha):
        """Blend the hybrid image with a brightness-matched fractal.

        Raw fractal images can be very dark or high contrast. Matching their
        channel statistics to the hybrid image before blending keeps the final
        image close to the original/generated content and lets lambda control
        style strength instead of overall brightness.
        """
        alpha = float(np.clip(alpha, 0.0, 0.15))
        overlay_img = fractal_img.convert("RGB").resize(base_img.size)
        base_array = np.array(base_img.convert("RGB"), dtype=np.float32)
        overlay_array = np.array(overlay_img, dtype=np.float32)

        base_mean = base_array.mean(axis=(0, 1), keepdims=True)
        base_std = base_array.std(axis=(0, 1), keepdims=True).clip(min=1.0)
        overlay_mean = overlay_array.mean(axis=(0, 1), keepdims=True)
        overlay_std = overlay_array.std(axis=(0, 1), keepdims=True).clip(min=1.0)
        overlay_array = (overlay_array - overlay_mean) / overlay_std * base_std + base_mean

        blended_array = (1 - alpha) * base_array + alpha * overlay_array
        return Image.fromarray(np.clip(blended_array, 0, 255).astype(np.uint8)), alpha

    def _compose_diffusemix(self, orig_pil, generated_pil, seed):
        mode = getattr(self.args, "diffusemix_augmentation_mode", "diffusemix")
        if mode == "direct":
            return generated_pil, {"augmentation_mode": "direct", "mix_lambda": 0.0}
        hybrid, vertical, split, keep_original_on_first_side = self._source_style_combine_images(orig_pil, generated_pil, seed)
        fractals = self._load_fractal_images()
        fractal_idx = int(seed) % len(fractals)
        requested_lam = float(getattr(self.args, "diffusemix_fractal_lambda", 0.08))
        augmented, lam = self._source_style_blend_fractal(hybrid, fractals[fractal_idx], requested_lam)
        meta = {
            "augmentation_mode": "diffusemix",
            "diffusemix_source_style": True,
            "diffusemix_use_real_fractal": bool(getattr(self.args, "diffusemix_use_real_fractal", True)),
            "mix_mask_type": "vertical" if vertical else "horizontal",
            "mix_mask_split": int(split),
            "mix_mask_original_first_side": bool(keep_original_on_first_side),
            "mix_lambda": lam,
            "mix_lambda_requested": requested_lam,
            "fractal_index": fractal_idx,
            "diffusemix_composition_version": "paper_quality_v3",
        }
        return augmented, meta

    @torch.no_grad()
    def _generate(self, pil, seed, prompt):
        pipe = self._ensure_pipe()
        gen = torch.Generator(device=self.device).manual_seed(int(seed)) if str(self.device).startswith("cuda") else torch.Generator().manual_seed(int(seed))
        out = pipe(prompt=prompt, image=pil, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7.5, generator=gen)
        return out.images[0]

    def _cam(self, algorithm, x, y):
        was_training = algorithm.training
        algorithm.eval()
        try:
            with torch.enable_grad():
                xi = x.detach().clone().requires_grad_(True)
                feat, inter = algorithm.featurizer(xi, ret_feats=True)
                logits = algorithm.classifier(feat)
                if int(y) < 0 or int(y) >= logits.shape[1]:
                    return None
                score = logits[:, int(y)].sum()
                algorithm.zero_grad(set_to_none=True)
                grads = torch.autograd.grad(score, inter[-1], retain_graph=False, create_graph=False)[0]
                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * inter[-1]).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
                cam = cam.flatten(1)
                cam = (cam - cam.min(1, keepdim=True).values) / (cam.max(1, keepdim=True).values - cam.min(1, keepdim=True).values + 1e-6)
                return cam.detach().view(1, 1, x.shape[-2], x.shape[-1])
        except Exception:
            return None
        finally:
            algorithm.train(was_training)

    @torch.no_grad()
    def _style_distance(self, x1, x2):
        # simple color-statistic distance; used only when style filter is enabled
        u1, s1 = x1.mean(dim=(2,3)), x1.std(dim=(2,3))
        u2, s2 = x2.mean(dim=(2,3)), x2.std(dim=(2,3))
        return torch.norm(torch.cat([u1-u2, s1-s2], dim=1), dim=1).item()

    def _filter(self, algorithm, orig_x, cand_x, y):
        meta = {"anchor_pred": None, "anchor_conf": None, "candidate_pred": None, "candidate_conf": None,
                "anchor_class_match": False, "candidate_class_match": False, "class_filter_pass": False,
                "conf_filter_pass": False, "kl_filter_pass": True, "cam_filter_pass": None,
                "foreground_filter_pass": None, "style_filter_pass": None, "kl_to_original": None,
                "cam_similarity": None, "mask_iou": None, "foreground_similarity": None,
                "style_distance": None, "style_gate": 1.0, "reliability_weight": 0.0,
                "filter_pass": False, "strong_positive": False, "positive_type": "invalid"}
        was_training = algorithm.training
        algorithm.eval()
        with torch.no_grad():
            lo = algorithm.predict(orig_x); lc = algorithm.predict(cand_x)
            if int(y) < 0 or int(y) >= lo.shape[1] or int(y) >= lc.shape[1]:
                meta["label_filter_pass"] = False
                algorithm.train(was_training); return meta
            meta["label_filter_pass"] = True
            po, pc = F.softmax(lo, dim=1), F.softmax(lc, dim=1)
            anchor_conf, anchor_pred = po.max(1)
            cand_conf, cand_pred = pc.max(1)
            meta["anchor_pred"] = int(anchor_pred.item()); meta["anchor_conf"] = float(anchor_conf.item())
            meta["candidate_pred"] = int(cand_pred.item()); meta["candidate_conf"] = float(cand_conf.item())
            meta["anchor_class_match"] = bool(anchor_pred.item() == int(y))
            meta["candidate_class_match"] = bool(cand_pred.item() == int(y))
            meta["class_filter_pass"] = bool(meta["anchor_class_match"] and meta["candidate_class_match"])
            meta["conf_filter_pass"] = bool(cand_conf.item() > self.args.diffusemix_filter_conf)
            meta["kl_to_original"] = float(F.kl_div(pc.log(), po, reduction="batchmean").item())
        if not meta["class_filter_pass"] or not meta["conf_filter_pass"]:
            algorithm.train(was_training); return meta
        if self.args.diffusemix_filter_kl is not None:
            meta["kl_filter_pass"] = bool(meta["kl_to_original"] < self.args.diffusemix_filter_kl)
            if not meta["kl_filter_pass"]:
                algorithm.train(was_training); return meta
        cam_ok, fg_ok = True, True
        cam_o = cam_c = None
        if self.args.diffusemix_use_cam_filter or self.args.diffusemix_use_fg_consistency:
            cam_o = self._cam(algorithm, orig_x, y); cam_c = self._cam(algorithm, cand_x, y)
            if cam_o is None or cam_c is None:
                cam_ok = False
                meta["cam_filter_pass"] = False
            else:
                meta["cam_similarity"] = float(F.cosine_similarity(cam_o.flatten(1), cam_c.flatten(1)).item())
                mo = cam_o > self.args.diffusemix_cam_threshold; mc = cam_c > self.args.diffusemix_cam_threshold
                inter = (mo & mc).float().sum(); union = (mo | mc).float().sum().clamp_min(1.0)
                meta["mask_iou"] = float((inter / union).item())
                cam_ok = (meta["cam_similarity"] > self.args.diffusemix_cam_sim_threshold or
                          meta["mask_iou"] > self.args.diffusemix_mask_iou_threshold)
                meta["cam_filter_pass"] = bool(cam_ok)
        with torch.no_grad():
            fo = algorithm.featurizer(orig_x); fc = algorithm.featurizer(cand_x)
            meta["foreground_similarity"] = float(F.cosine_similarity(fo, fc).mean().item())
        if self.args.diffusemix_semantic_sim_threshold is not None:
            fg_ok = meta["foreground_similarity"] > self.args.diffusemix_semantic_sim_threshold
            meta["foreground_filter_pass"] = bool(fg_ok)
        if self.args.diffusemix_use_style_filter:
            meta["style_distance"] = self._style_distance(orig_x, cand_x)
            meta["style_filter_pass"] = bool(meta["style_distance"] > self.args.diffusemix_style_min_distance)
            meta["style_gate"] = 1.0 if meta["style_filter_pass"] else 0.0
            if not meta["style_filter_pass"]:
                algorithm.train(was_training); return meta
        cam_sim_for_weight = meta["cam_similarity"] if meta["cam_similarity"] is not None else 1.0
        fg_sim_for_weight = meta["foreground_similarity"] if meta["foreground_similarity"] is not None else 0.0
        meta["reliability_weight"] = float(cand_conf.item() * cam_sim_for_weight * fg_sim_for_weight * meta["style_gate"])
        if self.args.diffusemix_use_reliability_gate and meta["reliability_weight"] < self.args.diffusemix_min_reliability:
            algorithm.train(was_training); return meta
        meta["filter_pass"] = True
        strong_by_metrics = (cand_conf.item() > self.args.diffusemix_strong_conf and cam_ok and fg_ok)
        if cam_o is not None and cam_c is not None:
            strong_by_metrics = strong_by_metrics and meta["cam_similarity"] > self.args.diffusemix_cam_sim_threshold and meta["mask_iou"] > self.args.diffusemix_mask_iou_threshold
        else:
            strong_by_metrics = False
        strong_by_reliability = (not self.args.diffusemix_use_reliability_gate or meta["reliability_weight"] >= self.args.diffusemix_strong_reliability)
        meta["strong_positive"] = bool(strong_by_metrics and strong_by_reliability)
        meta["positive_type"] = "strong" if meta["strong_positive"] else "weak"
        algorithm.train(was_training)
        return meta

    def _save(self, pil, meta, source_env, class_name, image_id):
        d = self._cache_dir(source_env, class_name, image_id); d.mkdir(parents=True, exist_ok=True)
        existing = sorted(d.glob("sample_*.json"))
        if len([p for p in existing if json.loads(p.read_text()).get("filter_pass", False)]) >= self.args.diffusemix_cache_max_per_image:
            return 0
        idx = len(existing)
        stem = f"sample_{idx:03d}"
        pil.save(d / f"{stem}.png")
        (d / f"{stem}.json").write_text(json.dumps(meta, indent=2))
        return 1

    def build_batch(self, algorithm, all_x, all_y, source_envs, paths=None, indices=None, step=0):
        zeros = {"loss_dm_fg": 0.0, "loss_dm_pair": 0.0, "number_of_diffusemix_positives_used_in_supcon": 0}
        if step < self.args.diffusemix_warmup_steps:
            return None, zeros
        xs=[]; anchors=[]; strong=[]; weights=[]; metas=[]
        generated = kept = invalid = weak = strong_n = saved = hit = miss = 0
        for i in range(all_x.shape[0]):
            if generated >= self.args.diffusemix_max_per_step and not self.args.diffusemix_use_cache_first:
                break
            if torch.rand(1).item() > self.args.diffusemix_generate_prob:
                continue
            y = int(all_y[i].item()); se = int(source_envs[i].item())
            if y < 0 or (self.class_names and y >= len(self.class_names)):
                invalid += 1
                self.stats["diffusemix_invalid_label_num"] += 1
                continue
            cn = self._class_name(y)
            p = paths[i] if paths else ""; iid = image_id_from_path(p, indices[i].item() if indices is not None else i)
            self.env_attempts[se] += 1; self.class_attempts[cn] += 1
            cached = self._load_cached(se, cn, iid) if self.args.diffusemix_use_cache_first else []
            if cached:
                hit += 1
                for img_path, meta in cached:
                    if int(meta.get("class_label", y)) != y or int(meta.get("source_env", se)) != se:
                        invalid += 1
                        self.stats["diffusemix_cache_label_mismatch_num"] += 1
                        continue
                    meta = dict(meta); meta["anchor_batch_index"] = int(i)
                    xs.append(self.basic(Image.open(img_path).convert("RGB"))); anchors.append(i); strong.append(bool(meta.get("strong_positive"))); weights.append(float(meta.get("reliability_weight", 1.0))); metas.append(meta)
                    kept += 1; strong_n += int(meta.get("strong_positive", False)); weak += int(not meta.get("strong_positive", False))
                    self.env_kept[se] += 1; self.class_kept[cn] += 1
                    if meta.get("strong_positive", False):
                        self.env_strong[se] += 1; self.class_strong[cn] += 1
                continue
            miss += 1
            if not self.args.diffusemix_regenerate_if_cache_empty or generated >= self.args.diffusemix_max_per_step:
                continue
            pil = Image.open(p).convert("RGB") if p else denorm_to_pil(all_x[i])
            seed = int(step * 100000 + i)
            selected_prompt = self.prompt(seed)
            with torch.no_grad():
                gen_pil = self._generate(pil, seed, selected_prompt)
            generated += 1
            gen_x = self.basic(gen_pil).unsqueeze(0).to(all_x.device)
            gen_meta = self._filter(algorithm, all_x[i:i+1], gen_x, y)
            if not gen_meta["filter_pass"]:
                invalid += 1
                self.stats["diffusemix_generated_quality_reject_num"] += 1
                continue
            cand_pil, mix_meta = self._compose_diffusemix(pil, gen_pil, seed)
            cand_x = self.basic(cand_pil).unsqueeze(0).to(all_x.device)
            meta = self._filter(algorithm, all_x[i:i+1], cand_x, y)
            meta.update({"dataset": self.dataset_name, "source_env": se, "class_name": cn, "class_label": y,
                         "original_path": p, "original_relpath": p, "prompt": selected_prompt, "seed": seed, "anchor_batch_index": int(i),
                         "generator_name": "instruct-pix2pix", "generated_quality_pass": True,
                         "generated_candidate_conf": gen_meta.get("candidate_conf"),
                         "generated_foreground_similarity": gen_meta.get("foreground_similarity"),
                         "created_at": datetime.utcnow().isoformat() + "Z", **mix_meta})
            if meta["filter_pass"]:
                kept += 1; strong_n += int(meta["strong_positive"]); weak += int(not meta["strong_positive"])
                self.env_kept[se] += 1; self.class_kept[cn] += 1
                if meta["strong_positive"]:
                    self.env_strong[se] += 1; self.class_strong[cn] += 1
                xs.append(cand_x.squeeze(0).detach().cpu()); anchors.append(i); strong.append(meta["strong_positive"]); weights.append(float(meta.get("reliability_weight", 1.0))); metas.append(meta)
                if self.args.diffusemix_save_kept_only or not self.args.diffusemix_save_rejected:
                    saved += self._save(cand_pil, meta, se, cn, iid)
            else:
                invalid += 1
                if self.args.diffusemix_save_rejected and not self.args.diffusemix_save_kept_only:
                    saved += self._save(cand_pil, meta, se, cn, iid)
        if not xs:
            stats = {**zeros, "diffusemix_cache_hit_num": hit, "diffusemix_cache_miss_num": miss, "diffusemix_generated_num": generated, "diffusemix_kept_num": kept, "diffusemix_strong_num": strong_n, "diffusemix_weak_num": weak, "diffusemix_invalid_num": invalid, "diffusemix_cache_save_num": saved}
            return None, self._rates(stats, metas)
        batch = {"x": torch.stack(xs).to(all_x.device), "anchor_indices": torch.tensor(anchors, device=all_x.device, dtype=torch.long), "strong_mask": torch.tensor(strong, device=all_x.device, dtype=torch.bool), "reliability_weights": torch.tensor(weights, device=all_x.device, dtype=torch.float32), "metas": metas}
        stats = {"diffusemix_cache_hit_num": hit, "diffusemix_cache_miss_num": miss, "diffusemix_generated_num": generated, "diffusemix_kept_num": kept, "diffusemix_strong_num": strong_n, "diffusemix_weak_num": weak, "diffusemix_invalid_num": invalid, "diffusemix_cache_save_num": saved}
        return batch, self._rates(stats, metas)

    def _rates(self, stats, metas):
        kept = stats.get("diffusemix_kept_num", 0); gen = stats.get("diffusemix_generated_num", 0)
        stats["diffusemix_keep_rate"] = float(kept / max(1, gen + stats.get("diffusemix_cache_hit_num", 0)))
        stats["diffusemix_strong_rate"] = float(stats.get("diffusemix_strong_num", 0) / max(1, kept))
        stats["diffusemix_invalid_label_num"] = int(self.stats.get("diffusemix_invalid_label_num", 0))
        stats["diffusemix_cache_label_mismatch_num"] = int(self.stats.get("diffusemix_cache_label_mismatch_num", 0))
        stats["diffusemix_generated_quality_reject_num"] = int(self.stats.get("diffusemix_generated_quality_reject_num", 0))
        stats["diffusemix_valid_num"] = int(stats.get("diffusemix_kept_num", 0))
        anchor_vals = [m.get("anchor_batch_index") for m in metas if m.get("anchor_batch_index") is not None]
        stats["diffusemix_anchor_min"] = int(min(anchor_vals)) if anchor_vals else -1
        stats["diffusemix_anchor_max"] = int(max(anchor_vals)) if anchor_vals else -1
        rel_vals = [m.get("reliability_weight") for m in metas if m.get("reliability_weight") is not None]
        stats["diffusemix_reliability_mean"] = float(sum(rel_vals) / len(rel_vals)) if rel_vals else 0.0
        for k, name in [("cam_similarity","diffusemix_cam_sim_mean"),("mask_iou","diffusemix_mask_iou_mean"),("foreground_similarity","diffusemix_fg_sim_mean"),("style_distance","diffusemix_style_distance_mean")]:
            vals=[m.get(k) for m in metas if m.get(k) is not None]
            stats[name]=float(sum(vals)/len(vals)) if vals else 0.0
        stats["kept_per_source_env"] = dict(self.env_kept)
        stats["attempts_per_source_env"] = dict(self.env_attempts)
        stats["keep_rate_per_source_env"] = {str(k): float(self.env_kept[k] / max(1, self.env_attempts[k])) for k in self.env_attempts}
        stats["strong_per_source_env"] = dict(self.env_strong)
        stats["strong_rate_per_source_env"] = {str(k): float(self.env_strong[k] / max(1, self.env_kept[k])) for k in self.env_attempts}
        stats["kept_per_class"] = dict(self.class_kept)
        stats["attempts_per_class"] = dict(self.class_attempts)
        stats["keep_rate_per_class"] = {str(k): float(self.class_kept[k] / max(1, self.class_attempts[k])) for k in self.class_attempts}
        stats["strong_per_class"] = dict(self.class_strong)
        stats["strong_rate_per_class"] = {str(k): float(self.class_strong[k] / max(1, self.class_kept[k])) for k in self.class_attempts}
        for key, field in [("class_filter_pass", "diffusemix_class_pass_num"), ("conf_filter_pass", "diffusemix_conf_pass_num"), ("kl_filter_pass", "diffusemix_kl_pass_num"), ("cam_filter_pass", "diffusemix_cam_pass_num"), ("foreground_filter_pass", "diffusemix_fg_pass_num"), ("style_filter_pass", "diffusemix_style_pass_num")]:
            vals = [m.get(key) for m in metas if m.get(key) is not None]
            stats[field] = int(sum(bool(v) for v in vals)) if vals else 0
        return stats
