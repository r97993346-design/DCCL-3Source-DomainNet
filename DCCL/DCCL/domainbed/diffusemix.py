import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

DEFAULT_PROMPT = (
    "Change only the domain-related appearance of this image, including texture, color, "
    "illumination, background, and artistic style, while strictly preserving the object "
    "identity, object shape, pose, and semantic category. Do not add or remove objects."
)

_basic = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_to_pil = T.ToPILImage()
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
        self.basic = _basic
        self.stats = defaultdict(float)
        self.env_attempts = defaultdict(int)
        self.env_kept = defaultdict(int)
        self.class_attempts = defaultdict(int)
        self.class_kept = defaultdict(int)

    def prompt(self):
        return DEFAULT_PROMPT

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

    @torch.no_grad()
    def _generate(self, pil, seed):
        pipe = self._ensure_pipe()
        gen = torch.Generator(device=self.device).manual_seed(int(seed)) if str(self.device).startswith("cuda") else torch.Generator().manual_seed(int(seed))
        out = pipe(prompt=self.prompt(), image=pil, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7.5, generator=gen)
        return out.images[0]

    def _cam(self, algorithm, x, y):
        was_training = algorithm.training
        algorithm.eval()
        try:
            with torch.enable_grad():
                xi = x.detach().clone().requires_grad_(True)
                feat, inter = algorithm.featurizer(xi, ret_feats=True)
                logits = algorithm.classifier(feat)
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
        meta = {"anchor_pred": None, "anchor_conf": None, "kl_to_original": None,
                "cam_similarity": None, "mask_iou": None, "foreground_similarity": None,
                "style_distance": None, "filter_pass": False, "strong_positive": False}
        was_training = algorithm.training
        algorithm.eval()
        with torch.no_grad():
            lo = algorithm.predict(orig_x); lc = algorithm.predict(cand_x)
            po, pc = F.softmax(lo, dim=1), F.softmax(lc, dim=1)
            conf, pred = pc.max(1)
            meta["anchor_pred"] = int(pred.item()); meta["anchor_conf"] = float(conf.item())
            meta["kl_to_original"] = float(F.kl_div(pc.log(), po, reduction="batchmean").item())
        if pred.item() != int(y) or conf.item() <= self.args.diffusemix_filter_conf:
            algorithm.train(was_training); return meta
        if self.args.diffusemix_filter_kl is not None and meta["kl_to_original"] >= self.args.diffusemix_filter_kl:
            algorithm.train(was_training); return meta
        cam_ok, fg_ok = True, True
        cam_o = cam_c = None
        if self.args.diffusemix_use_cam_filter or self.args.diffusemix_use_fg_consistency:
            cam_o = self._cam(algorithm, orig_x, y); cam_c = self._cam(algorithm, cand_x, y)
            if cam_o is None or cam_c is None:
                cam_ok = False
            else:
                meta["cam_similarity"] = float(F.cosine_similarity(cam_o.flatten(1), cam_c.flatten(1)).item())
                mo = cam_o > self.args.diffusemix_cam_threshold; mc = cam_c > self.args.diffusemix_cam_threshold
                inter = (mo & mc).float().sum(); union = (mo | mc).float().sum().clamp_min(1.0)
                meta["mask_iou"] = float((inter / union).item())
                cam_ok = (meta["cam_similarity"] > self.args.diffusemix_cam_sim_threshold or
                          meta["mask_iou"] > self.args.diffusemix_mask_iou_threshold)
        with torch.no_grad():
            fo = algorithm.featurizer(orig_x); fc = algorithm.featurizer(cand_x)
            meta["foreground_similarity"] = float(F.cosine_similarity(fo, fc).mean().item())
        if self.args.diffusemix_semantic_sim_threshold is not None:
            fg_ok = meta["foreground_similarity"] > self.args.diffusemix_semantic_sim_threshold
        if self.args.diffusemix_use_style_filter:
            meta["style_distance"] = self._style_distance(orig_x, cand_x)
            if meta["style_distance"] <= self.args.diffusemix_style_min_distance:
                algorithm.train(was_training); return meta
        meta["filter_pass"] = True
        meta["strong_positive"] = bool(conf.item() > self.args.diffusemix_strong_conf and cam_ok and fg_ok)
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
        xs=[]; anchors=[]; strong=[]; metas=[]
        generated = kept = invalid = weak = strong_n = saved = hit = miss = 0
        for i in range(all_x.shape[0]):
            if generated >= self.args.diffusemix_max_per_step and not self.args.diffusemix_use_cache_first:
                break
            if torch.rand(1).item() > self.args.diffusemix_generate_prob:
                continue
            y = int(all_y[i].item()); se = int(source_envs[i].item()); cn = self._class_name(y)
            p = paths[i] if paths else ""; iid = image_id_from_path(p, indices[i].item() if indices is not None else i)
            self.env_attempts[se] += 1; self.class_attempts[cn] += 1
            cached = self._load_cached(se, cn, iid) if self.args.diffusemix_use_cache_first else []
            if cached:
                hit += 1
                for img_path, meta in cached:
                    xs.append(self.basic(Image.open(img_path).convert("RGB"))); anchors.append(i); strong.append(bool(meta.get("strong_positive"))); metas.append(meta)
                    kept += 1; strong_n += int(meta.get("strong_positive", False)); weak += int(not meta.get("strong_positive", False))
                    self.env_kept[se] += 1; self.class_kept[cn] += 1
                continue
            miss += 1
            if not self.args.diffusemix_regenerate_if_cache_empty or generated >= self.args.diffusemix_max_per_step:
                continue
            pil = Image.open(p).convert("RGB") if p else denorm_to_pil(all_x[i])
            with torch.no_grad():
                cand_pil = self._generate(pil, int(step * 100000 + i))
            generated += 1
            cand_x = self.basic(cand_pil).unsqueeze(0).to(all_x.device)
            meta = self._filter(algorithm, all_x[i:i+1], cand_x, y)
            meta.update({"dataset": self.dataset_name, "source_env": se, "class_name": cn, "class_label": y,
                         "original_path": p, "original_relpath": p, "prompt": self.prompt(), "seed": int(step*100000+i),
                         "generator_name": "instruct-pix2pix", "created_at": datetime.utcnow().isoformat() + "Z"})
            if meta["filter_pass"]:
                kept += 1; strong_n += int(meta["strong_positive"]); weak += int(not meta["strong_positive"])
                self.env_kept[se] += 1; self.class_kept[cn] += 1
                xs.append(cand_x.squeeze(0).detach().cpu()); anchors.append(i); strong.append(meta["strong_positive"]); metas.append(meta)
                if self.args.diffusemix_save_kept_only or not self.args.diffusemix_save_rejected:
                    saved += self._save(cand_pil, meta, se, cn, iid)
            else:
                invalid += 1
                if self.args.diffusemix_save_rejected and not self.args.diffusemix_save_kept_only:
                    saved += self._save(cand_pil, meta, se, cn, iid)
        if not xs:
            stats = {**zeros, "diffusemix_cache_hit_num": hit, "diffusemix_cache_miss_num": miss, "diffusemix_generated_num": generated, "diffusemix_kept_num": kept, "diffusemix_strong_num": strong_n, "diffusemix_weak_num": weak, "diffusemix_invalid_num": invalid, "diffusemix_cache_save_num": saved}
            return None, self._rates(stats, metas)
        batch = {"x": torch.stack(xs).to(all_x.device), "anchor_indices": torch.tensor(anchors, device=all_x.device, dtype=torch.long), "strong_mask": torch.tensor(strong, device=all_x.device, dtype=torch.bool), "metas": metas}
        stats = {"diffusemix_cache_hit_num": hit, "diffusemix_cache_miss_num": miss, "diffusemix_generated_num": generated, "diffusemix_kept_num": kept, "diffusemix_strong_num": strong_n, "diffusemix_weak_num": weak, "diffusemix_invalid_num": invalid, "diffusemix_cache_save_num": saved}
        return batch, self._rates(stats, metas)

    def _rates(self, stats, metas):
        kept = stats.get("diffusemix_kept_num", 0); gen = stats.get("diffusemix_generated_num", 0)
        stats["diffusemix_keep_rate"] = float(kept / max(1, gen + stats.get("diffusemix_cache_hit_num", 0)))
        stats["diffusemix_strong_rate"] = float(stats.get("diffusemix_strong_num", 0) / max(1, kept))
        for k, name in [("cam_similarity","diffusemix_cam_sim_mean"),("mask_iou","diffusemix_mask_iou_mean"),("foreground_similarity","diffusemix_fg_sim_mean"),("style_distance","diffusemix_style_distance_mean")]:
            vals=[m.get(k) for m in metas if m.get(k) is not None]
            stats[name]=float(sum(vals)/len(vals)) if vals else 0.0
        stats["keep_rate_per_source_env"] = dict(self.env_kept)
        stats["strong_rate_per_source_env"] = dict(self.env_kept)
        stats["keep_rate_per_class"] = dict(self.class_kept)
        stats["strong_rate_per_class"] = dict(self.class_kept)
        return stats
