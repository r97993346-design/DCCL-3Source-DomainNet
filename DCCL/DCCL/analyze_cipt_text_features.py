#!/usr/bin/env python3
"""Compare paired B5B/B5A pk and zk using a fixed trained CIPT checkpoint.

Run from the directory containing train_all.py. --pk-only needs no checkpoint.
No training, template sampling, prompt selection, or model update is performed.
"""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
from cipt_feature_report import save_report, write_csv


DEFAULT_CLASSES = {
    "PACS": ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
    "VLCS": ["bird", "car", "chair", "dog", "person"],
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}


def load_module(name, path):
    # Import the actual branch modules without loading DomainBed's full trainer.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_name(name):
    return str(name).replace("_", " ").strip().lower()


def checkpoint_state(checkpoint):
    state = checkpoint.get("model_dict", checkpoint.get("state_dict", checkpoint))
    suffix = "causal_decomposition.causal_adapter.weight"
    matches = [key for key in state if key.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError("Checkpoint must contain exactly one CIPT causal adapter; got {}.".format(matches))
    prefix = matches[0][:-len(suffix)]
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def substate(state, prefix):
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def resolve_data_root(data_dir, dataset):
    if not data_dir:
        raise ValueError("--data-dir is required to extract zk from images.")
    base = Path(data_dir).expanduser().resolve()
    for candidate in [base / dataset, base]:
        if candidate.is_dir() and any(child.is_dir() and not child.name.startswith(".")
                                     for child in candidate.iterdir()):
            if candidate == base / dataset or base.name.lower() == dataset.lower():
                return candidate
    raise FileNotFoundError("Cannot find {} image directory under {}. Pass its parent or the dataset directory.".format(dataset, base))


def selected_class_ids(class_names, requested):
    lookup = {canonical_name(name): i for i, name in enumerate(class_names)}
    ids = []
    for name in requested:
        if canonical_name(name) not in lookup:
            raise ValueError("Unknown class {!r}; available classes: {}".format(name, class_names))
        ids.append(lookup[canonical_name(name)])
    if len(set(ids)) != len(ids) or len(ids) < 2:
        raise ValueError("Choose at least two distinct classes for class-distance diagnostics.")
    return ids


def select_samples(data_root, class_names, class_ids, domains, count, seed, manifest=None):
    """One fixed manifest is reused for both text conditions."""
    if manifest:
        with Path(manifest).open(encoding="utf-8-sig", newline="") as f:
            old_rows = list(csv.DictReader(f))
        rows = []
        for row in old_rows:
            if int(row["class_id"]) not in class_ids:
                raise ValueError("Manifest contains a class outside --classes.")
            relative = Path(row["relative_path"])
            path = (data_root / relative).resolve()
            if not path.is_file() or data_root not in path.parents:
                raise ValueError("Manifest image missing or outside data directory: {}".format(path))
            c = int(row["class_id"])
            if canonical_name(row["class"]) != canonical_name(class_names[c]):
                raise ValueError("Manifest class order differs from checkpoint class order.")
            rows.append({"image_id": len(rows), "class_id": c, "class": class_names[c],
                         "domain": row["domain"], "relative_path": str(relative), "path": str(path)})
        if not rows or set(r["class_id"] for r in rows) != set(class_ids):
            raise ValueError("Manifest must include all selected classes.")
        return rows
    available = sorted(d.name for d in data_root.iterdir()
                       if d.is_dir() and not d.name.startswith("."))
    domains = domains or (["photo", "sketch"] if {"photo", "sketch"}.issubset(available)
                          else list(dict.fromkeys([available[0], available[-1]])))
    if len(set(domains)) != len(domains) or not set(domains).issubset(available):
        raise ValueError("Choose distinct existing domains; available: {}".format(available))
    rng = np.random.default_rng(seed)
    rows = []
    for domain in domains:
        folders = {canonical_name(p.name): p for p in (data_root / domain).iterdir() if p.is_dir()}
        if set(folders) != {canonical_name(n) for n in class_names}:
            raise ValueError("Dataset classes in {} do not match checkpoint classes: {} vs {}".format(
                domain, sorted(folders), class_names))
        for c in class_ids:
            folder = folders[canonical_name(class_names[c])]
            paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
            if len(paths) < count:
                raise ValueError("Need {} images in {}, found {}.".format(count, folder, len(paths)))
            chosen = np.sort(rng.choice(len(paths), count, replace=False))
            for j in chosen:
                path = paths[int(j)].resolve()
                rows.append({"image_id": len(rows), "class_id": c, "class": class_names[c],
                             "domain": domain, "relative_path": str(path.relative_to(data_root)), "path": str(path)})
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULT_CLASSES), default="PACS")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir")
    parser.add_argument("--checkpoint", help="A trained DomainBed CIPTDCCL TE*.pth checkpoint.")
    parser.add_argument("--clip-path", "--cipt_clip_path", dest="clip_path", help="Local OpenAI CLIP .pt file; no download is performed.")
    parser.add_argument("--classes", nargs="+", help="Default: dog horse guitar for PACS; bird car person for VLCS.")
    parser.add_argument("--domains", nargs="+", help="Exact dataset folder names; defaults to photo+sketch if present, else first+last.")
    parser.add_argument("--samples-per-class-domain", type=int, default=5)
    parser.add_argument("--sample-manifest", help="Reuse a previous samples.csv to fix images across checkpoints.")
    parser.add_argument("--neutral-subject", choices=["subject", "thing", "object", "entity"], default="subject")
    parser.add_argument("--pk-only", action="store_true", help="Analyze frozen text features only; no random TDA is used.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tsne-perplexity", type=float, default=30)
    parser.add_argument("--tsne-templates-per-image", type=int, default=80,
                        help="Plotting-only cap (1..80); every metric always uses all 80 texts.")
    parser.add_argument("--no-tsne", action="store_true")
    parser.add_argument("--output-dir", default="train_output/ablation/pk_zk_paired_80")
    args = parser.parse_args()
    for name in ["samples_per_class_domain", "batch_size", "text_batch_size", "tsne_templates_per_image"]:
        if getattr(args, name) < 1:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.tsne_templates_per_image > 80 or not np.isfinite(args.tsne_perplexity) or args.tsne_perplexity <= 0:
        parser.error("t-SNE template count must be <=80 and perplexity must be finite and positive.")
    if not args.pk_only and not args.checkpoint:
        parser.error("zk requires --checkpoint; use --pk-only for checkpoint-free text diagnostics.")
    if args.sample_manifest and args.pk_only:
        parser.error("--sample-manifest applies to image-based zk diagnostics.")
    return args


def main():
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn.functional as F
    from PIL import Image

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run on the training server or pass --device cpu.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    checkpoint, hparams, state = None, {}, None
    if args.checkpoint:
        # These are the user's own training checkpoints, including args/hparams.
        kwargs = {"map_location": "cpu"}
        import inspect
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False
        checkpoint = torch.load(Path(args.checkpoint).expanduser(), **kwargs)
        hparams = checkpoint.get("model_hparams", checkpoint.get("hparams", {}))
        state = checkpoint_state(checkpoint)
        trained_dataset = checkpoint.get("args", {}).get("dataset")
        if trained_dataset and trained_dataset != args.dataset:
            raise ValueError("Checkpoint dataset {} != --dataset {}".format(trained_dataset, args.dataset))
        if not args.pk_only and hparams.get("cipt_use_tda", True) in (False, "false", "False", 0):
            raise ValueError("This checkpoint was trained with TDA disabled; choose a TDA-trained checkpoint for zk.")
    data_root = resolve_data_root(args.data_dir, args.dataset) if not args.pk_only else None
    class_names = hparams.get("cipt_class_names")
    if not class_names and data_root is not None:
        first_domain = sorted(d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith("."))[0]
        class_names = sorted(d.name for d in first_domain.iterdir() if d.is_dir())
    class_names = list(class_names or DEFAULT_CLASSES[args.dataset])
    if any(canonical_name(n).startswith("class ") for n in class_names):
        raise ValueError("Checkpoint uses anonymous class prompts; real class names are required for this experiment.")
    requested = args.classes or (["dog", "horse", "guitar"] if args.dataset == "PACS" else ["bird", "car", "person"])
    class_ids = selected_class_ids(class_names, requested)
    clip_path = args.clip_path or hparams.get("cipt_clip_path", "")
    if not clip_path or not Path(clip_path).expanduser().is_file():
        raise FileNotFoundError("Pass --clip-path pointing to the local CLIP checkpoint used for training.")

    prompt = load_module("_cipt_diagnostic_prompt", ROOT / "domainbed/algorithms/cipt_prompt.py")
    modules = load_module("_cipt_diagnostic_modules", ROOT / "domainbed/algorithms/cipt_modules.py")
    model, tokenize = prompt.load_frozen_clip(hparams.get("cipt_clip_backbone", "ViT-B/16"), clip_path)
    clip_state = substate(state, "clip_model.") if state is not None else {}
    if clip_state:
        model.load_state_dict(clip_state, strict=True)
    model.to(args.device).eval().requires_grad_(False)
    dim = int(model.visual.output_dim)
    class_texts = [template.format(class_names[c].replace("_", " "))
                   for c in class_ids for template in prompt.B5B_IMAGENET_TEMPLATES]
    agnostic_texts = list(prompt.build_paired_generic_templates(args.neutral_subject))
    if len(agnostic_texts) != 80 or len(class_texts) != 80 * len(class_ids):
        raise ValueError("The branch's paired bank is not 80 texts.")

    def encode_texts(texts):
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), args.text_batch_size):
                tokens = tokenize(texts[start:start + args.text_batch_size]).to(args.device)
                chunks.append(model.encode_text(tokens).float())
        raw = torch.cat(chunks)
        return raw, F.normalize(raw, dim=-1)

    pc_raw, pc = encode_texts(class_texts)
    pc_raw, pc = pc_raw.reshape(len(class_ids), 80, dim), pc.reshape(len(class_ids), 80, dim)
    pa_raw, pa = encode_texts(agnostic_texts)
    features = {"pk_raw_class": pc_raw.cpu().numpy(), "pk_class": pc.cpu().numpy(),
                "pk_raw_agnostic": pa_raw.cpu().numpy(), "pk_agnostic": pa.cpu().numpy()}
    metadata = {
        "dataset": args.dataset, "seed": args.seed,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else None,
        "checkpoint_test_envs": checkpoint.get("test_envs") if checkpoint is not None else None,
        "checkpoint_training_template_mode": hparams.get("cipt_template_mode"),
        "checkpoint_hparams": hparams,
        "clip_path": str(Path(clip_path).expanduser().resolve()),
        "clip_weights_loaded_from_checkpoint": bool(clip_state),
        "comparison": "B5B class-conditioned vs B5A paired neutral-subject; fixed weights",
        "neutral_subject": args.neutral_subject, "template_count": 80,
        "pk_definition": "F.normalize(CLIP.encode_text(environment_text), dim=-1)",
        "zk_definition": "actual TextDiversityAugmentation(E, pk) output after LayerNorm, before L2 normalization",
        "tsne_perplexity": args.tsne_perplexity,
        "tsne_templates_per_image": args.tsne_templates_per_image,
        "pk_only": args.pk_only, "device": args.device,
        "numpy_version": np.__version__, "torch_version": str(torch.__version__),
    }
    # Ensure uncommon checkpoint hparam scalar types remain serializable.
    metadata = json.loads(json.dumps(metadata, default=str))
    samples, labels, domains = None, None, None
    if not args.pk_only:
        decomposition = modules.CausalDecomposition(dim)
        decomposition.load_state_dict(substate(state, "causal_decomposition."), strict=True)
        tda = modules.TextDiversityAugmentation(dim, int(hparams.get("cipt_tda_heads", 1)))
        tda.load_state_dict(substate(state, "tda."), strict=True)
        decomposition.to(args.device).eval().requires_grad_(False)
        tda.to(args.device).eval().requires_grad_(False)
        transforms = load_module("_cipt_diagnostic_transforms", ROOT / "domainbed/datasets/transforms.py")
        samples = select_samples(data_root, class_names, class_ids, args.domains,
                                 args.samples_per_class_domain, args.seed, args.sample_manifest)
        labels = np.asarray([row["class_id"] for row in samples], dtype=int)
        domains = np.asarray([row["domain"] for row in samples])
        metadata["image_count"] = len(samples)
        metadata["domains"] = sorted(set(domains.tolist()))
        metadata["preprocessing"] = "branch clip_basic: bicubic Resize(224), CenterCrop(224), RGB, CLIP normalize"
        collected = {key: [] for key in ["visual_raw", "visual", "e", "zk_class", "zk_agnostic",
                                       "zk_preln_class", "zk_preln_agnostic",
                                       "tda_update_class", "tda_update_agnostic"]}
        captured = []

        def capture_preln(module, inputs):
            captured.append(inputs[0].detach())

        hook = tda.layer_norm.register_forward_pre_hook(capture_preln)
        try:
            with torch.inference_mode():
                for start in range(0, len(samples), args.batch_size):
                    current = samples[start:start + args.batch_size]
                    tensors = []
                    for row in current:
                        with Image.open(row["path"]) as image:
                            tensors.append(transforms.clip_basic(image))
                    images = torch.stack(tensors).to(args.device)
                    visual_raw = model.encode_image(images).float()
                    visual = F.normalize(visual_raw, dim=-1)
                    e, _ = decomposition(visual)
                    local_ids = torch.tensor([class_ids.index(row["class_id"]) for row in current], device=args.device)
                    contexts = {"class": pc[local_ids], "agnostic": pa}
                    collected["visual_raw"].append(visual_raw.cpu().numpy())
                    collected["visual"].append(visual.cpu().numpy())
                    collected["e"].append(e.cpu().numpy())
                    for condition, bank in contexts.items():
                        captured.clear()
                        z = tda(e, bank)
                        if len(captured) != 1:
                            raise RuntimeError("Expected one LayerNorm invocation inside the branch's TDA.")
                        preln = captured[0].reshape(len(current), 80, dim)
                        collected["zk_" + condition].append(z.cpu().numpy())
                        collected["zk_preln_" + condition].append(preln.cpu().numpy())
                        collected["tda_update_" + condition].append((preln - e[:, None, :]).cpu().numpy())
                    print("Extracted {}/{} images, 80 paired texts each".format(
                        min(start + args.batch_size, len(samples)), len(samples)), flush=True)
        finally:
            hook.remove()
        features.update({key: np.concatenate(chunks) for key, chunks in collected.items()})
        # Explicitly save the normalized zk consumed by cosine classification.
        from cipt_feature_report import unit
        features["zk_unit_class"] = unit(features["zk_class"]).astype(np.float32)
        features["zk_unit_agnostic"] = unit(features["zk_agnostic"]).astype(np.float32)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    text_rows = [{"condition": "B5B", "class": class_names[c], "template_id": k,
                  "text": class_texts[local * 80 + k]}
                 for local, c in enumerate(class_ids) for k in range(80)]
    text_rows += [{"condition": "B5A", "class": "shared", "template_id": k, "text": text}
                 for k, text in enumerate(agnostic_texts)]
    write_csv(out / "texts.csv", text_rows)
    print("Computing statistics and joint plots in {}".format(out), flush=True)
    save_report(out, features, class_names, class_ids, metadata, labels, domains, samples,
                make_tsne=not args.no_tsne)
    print("Done. Read report.md, summary.json, pk_paired.csv and zk_image_means.csv.")


if __name__ == "__main__":
    main()
