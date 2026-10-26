#!/usr/bin/env python3
"""Diagnose trained CIPT B5B/B5A E and native text-fused Z representations.

Uses actual branch modules and a trained legacy B5B or paired B5A checkpoint.
No training or label-free performance evaluation occurs. See CIPT_MECHANISM_VISUALIZATION.md.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
from analyze_cipt_text_features import (DEFAULT_CLASSES, canonical_name, checkpoint_state,
                                       load_module, resolve_data_root, select_samples,
                                       selected_class_ids, substate)
from cipt_mechanism_report import (dump_json, plot_metrics, plot_overview_tsne,
                                   plot_template_tsne, plot_tsne, summarize, write_csv)


def as_bool(value):
    if isinstance(value, str):
        if value.lower() not in {"true", "false"}:
            raise ValueError("Invalid boolean: {}".format(value))
        return value.lower() == "true"
    return bool(value)


def validate_original_checkpoint(hparams):
    if not hparams:
        raise ValueError("Checkpoint lacks training hparams; original CIPT provenance cannot be verified.")
    if hparams.get("cipt_objective", "legacy") != "legacy":
        raise ValueError("Choose a legacy CIPT checkpoint; RD/MCC/env_anchor/env_kl is a different objective.")
    mode = str(hparams.get("cipt_template_mode", "")).lower()
    if mode not in {"b5b", "b5a"}:
        raise ValueError("Choose a diverse B5B or paired 80-template B5A checkpoint.")
    if hparams.get("cipt_selector_mode", "random") != "random":
        raise ValueError("Use random-K training for the paired B5B/B5A comparison; added selectors differ.")
    for name in ["cipt_use_tda", "cipt_use_de", "cipt_use_ind"]:
        if not as_bool(hparams.get(name, True)):
            raise ValueError("Original CIPT requires {}=true during training.".format(name))
    if as_bool(hparams.get("cipt_use_contrastive", True)) and not as_bool(hparams.get("cipt_pure", False)):
        raise ValueError("Checkpoint includes added contrastive training; use the original-CIPT preset.")
    if as_bool(hparams.get("cipt_use_aug_view", False)):
        raise ValueError("Choose the branch's original single-view CIPT checkpoint.")
    return mode


def restore_text_bank(state, key, encoded, device, required=False):
    """Verify template/subject/class mapping, then use the exact saved bank.

    Re-encoding alone cannot establish that an older B5A checkpoint trained on
    the current paired 80-template bank, so its native bank is required.
    """
    import torch
    import torch.nn.functional as F
    saved = state.get(key)
    if saved is None:
        if required:
            raise ValueError("Checkpoint lacks {}; cannot verify paired B5A training texts.".format(key))
        return encoded
    if tuple(saved.shape) != tuple(encoded.shape):
        raise ValueError("Saved {} shape differs from current classes/templates/subject.".format(key))
    saved = saved.to(device).float()
    if not torch.isfinite(saved).all() or torch.any(saved.norm(dim=-1) < 1e-12):
        raise ValueError("Saved {} contains invalid text vectors.".format(key))
    drift = 1 - (F.normalize(saved, dim=-1) * encoded).sum(-1).clamp(-1, 1)
    if not torch.isfinite(drift).all() or float(drift.max()) > 1e-4:
        raise ValueError("Saved {} differs from current class/template/subject/CLIP mapping.".format(key))
    return saved


def validate_samples(samples, class_ids, domains):
    paths = [row["relative_path"] for row in samples]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate images in manifest would bias paired statistics.")
    if set(row["domain"] for row in samples) != set(domains):
        raise ValueError("Manifest domains differ from --domains (default: every dataset domain).")
    for row in samples:
        parts = Path(row["relative_path"]).parts
        if len(parts) < 3 or parts[0] != row["domain"] or canonical_name(parts[1]) != canonical_name(row["class"]):
            raise ValueError("Manifest domain/class labels disagree with the dataset folder path.")
    counts = [sum(row["class_id"] == c and row["domain"] == d for row in samples)
              for c in class_ids for d in domains]
    if min(counts) < 2 or len(set(counts)) != 1:
        raise ValueError("Use equal counts (>=2) in every selected class/domain cell.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULT_CLASSES), default="PACS")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-dir")
    parser.add_argument("--clip-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classes", nargs="+", help="Default: all classes.")
    parser.add_argument("--domains", nargs="+", help="Default: all domains, including the held-out domain.")
    parser.add_argument("--samples-per-class-domain", type=int, default=25)
    parser.add_argument("--sample-manifest", help="Reuse samples.csv from a previous extraction.")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--report-only", action="store_true", help="Reuse extracted arrays; no PyTorch/CLIP needed.")
    mode.add_argument("--extract-only", action="store_true")
    parser.add_argument("--no-tsne", action="store_true")
    parser.add_argument("--tsne-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--perplexities", nargs="+", type=float, default=[10, 30])
    parser.add_argument("--template-id", type=int, default=0, help="Fixed 0-based template for plots; metrics always use all 80.")
    parser.add_argument("--template-ids", nargs="+", type=int, default=[0, 20, 40, 60],
                        help="Preselected templates for the paired diversity plot; metrics still use all 80.")
    parser.add_argument("--skip-template-tsne", action="store_true",
                        help="Skip template comparison plots when only replotting domain structure.")
    parser.add_argument("--overview", action="store_true",
                        help="Also plot all selected classes: class colors and domain markers.")
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args(argv)
    if not args.report_only and (not args.checkpoint or not args.data_dir):
        parser.error("Extraction requires --checkpoint and --data-dir.")
    if args.samples_per_class_domain < 2 or min(args.batch_size, args.text_batch_size, args.neighbors) < 1:
        parser.error("Need >=2 images per class/domain and positive batch sizes/neighbors.")
    if not 0 <= args.template_id < 80 or args.tsne_iterations < 250:
        parser.error("Use template ID 0..79 and >=250 t-SNE iterations.")
    if (len(args.template_ids) < 2 or len(set(args.template_ids)) != len(args.template_ids)
            or any(not 0 <= k < 80 for k in args.template_ids)):
        parser.error("Use at least two distinct --template-ids in 0..79.")
    if args.pca_dim < 0 or args.pca_dim == 1 or any(not np.isfinite(p) or p <= 0 for p in args.perplexities):
        parser.error("PCA dimension must be 0 or >=2; perplexities must be finite and positive.")
    return args


def extract(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn.functional as F
    from PIL import Image

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run on the training server or use --device cpu.")
    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.sample_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False  # Trusted local training checkpoint, includes hparams/args.
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, **kwargs)
    hparams = checkpoint.get("model_hparams", checkpoint.get("hparams", {}))
    template_mode = validate_original_checkpoint(hparams)
    class_conditioned = template_mode == "b5b"
    neutral_subject = str(hparams.get("cipt_neutral_subject", "subject"))
    trained_dataset = checkpoint.get("args", {}).get("dataset")
    if trained_dataset and trained_dataset != args.dataset:
        raise ValueError("Checkpoint dataset differs from --dataset.")
    state = checkpoint_state(checkpoint)
    data_root = resolve_data_root(args.data_dir, args.dataset)
    available_domains = sorted(d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith("."))
    class_names = list(hparams.get("cipt_class_names") or sorted(
        d.name for d in (data_root / available_domains[0]).iterdir() if d.is_dir()))
    if any(canonical_name(name).startswith("class ") for name in class_names):
        raise ValueError("Real class names are required.")
    class_ids = selected_class_ids(class_names, args.classes or class_names)
    domains = args.domains or available_domains
    if len(set(domains)) < 2:
        raise ValueError("Select at least two domains for within-class domain analysis.")
    samples = select_samples(data_root, class_names, class_ids, domains,
                             args.samples_per_class_domain, args.sample_seed, args.sample_manifest)
    validate_samples(samples, class_ids, domains)
    out = Path(args.output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Extraction output must be empty. Use a new directory or --report-only.")
    out.mkdir(parents=True, exist_ok=True)
    clip_path = Path(args.clip_path or hparams.get("cipt_clip_path", "")).expanduser()
    if not clip_path.is_file():
        raise FileNotFoundError("Pass --clip-path for the local OpenAI CLIP checkpoint used in training.")
    prompt = load_module("_mechanism_prompt", ROOT / "domainbed/algorithms/cipt_prompt.py")
    modules = load_module("_mechanism_modules", ROOT / "domainbed/algorithms/cipt_modules.py")
    transforms = load_module("_mechanism_transforms", ROOT / "domainbed/datasets/transforms.py")
    model, tokenize = prompt.load_frozen_clip(hparams.get("cipt_clip_backbone", "ViT-B/16"), str(clip_path))
    clip_state = substate(state, "clip_model.")
    if clip_state:
        model.load_state_dict(clip_state, strict=True)
    model.to(args.device).eval().requires_grad_(False)
    dim = int(model.visual.output_dim)
    decomposition = modules.CausalDecomposition(dim)
    decomposition.load_state_dict(substate(state, "causal_decomposition."), strict=True)
    tda = modules.TextDiversityAugmentation(dim, int(hparams.get("cipt_tda_heads", 1)))
    tda.load_state_dict(substate(state, "tda."), strict=True)
    decomposition.to(args.device).eval().requires_grad_(False)
    tda.to(args.device).eval().requires_grad_(False)
    templates = list(prompt.B5B_IMAGENET_TEMPLATES)
    if len(templates) != 80:
        raise ValueError("Expected the original 80 B5B templates.")
    class_texts = [template.format(name.replace("_", " ")) for name in class_names for template in templates]
    neutral_texts = list(prompt.build_paired_generic_templates(neutral_subject))
    constant_subjects = class_names if class_conditioned else [neutral_subject]
    constant_texts = ["a photo of a {}.".format(name.replace("_", " ")) for name in constant_subjects]

    def encode(texts):
        with torch.inference_mode():
            return F.normalize(torch.cat([model.encode_text(tokenize(texts[i:i + args.text_batch_size]).to(args.device)).float()
                                           for i in range(0, len(texts), args.text_batch_size)]), dim=-1)

    pc = encode(class_texts).reshape(len(class_names), 80, dim)
    pn, pconst = encode(neutral_texts), encode(constant_texts)
    pc = restore_text_bank(state, "text_features.b5b_text_bank", pc, args.device)
    pn = restore_text_bank(state, "text_features.b5a_text_bank", pn, args.device,
                           required=not class_conditioned)
    native_bank_key = "text_features.{}_text_bank".format(template_mode)
    np.savez_compressed(out / "text_banks.npz", class_unit=pc.cpu().numpy(),
                        neutral_unit=pn.cpu().numpy(), constant_unit=pconst.cpu().numpy())
    text_rows = [{"condition": "class", "class": name, "template_id": k,
                  "native_z_bank": class_conditioned, "text": class_texts[c * 80 + k]}
                 for c, name in enumerate(class_names) for k in range(80)]
    text_rows += [{"condition": "neutral", "class": neutral_subject, "template_id": k,
                   "native_z_bank": not class_conditioned, "text": text}
                  for k, text in enumerate(neutral_texts)]
    text_rows += [{"condition": "constant", "class": name, "template_id": -1,
                   "native_z_bank": False, "text": constant_texts[c]}
                  for c, name in enumerate(constant_subjects)]
    write_csv(out / "texts.csv", text_rows)
    write_csv(out / "samples.csv", samples)
    z_key = "z_class_unit" if class_conditioned else "z_neutral_unit"
    arrays = {key: np.lib.format.open_memmap(out / (key + ".npy"), mode="w+", dtype=np.float32,
                                            shape=((len(samples), 80, dim) if key == z_key else (len(samples), dim)))
              for key in ["e_unit", "ln_e_unit", "z_constant_unit", z_key]}
    probe_keys = (["class_swap_cosine_shift", "neutral_swap_cosine_shift"] if class_conditioned
                  else ["class_insertion_cosine_shift"])
    probes = {key: np.zeros(len(samples)) for key in probe_keys}
    try:
        with torch.inference_mode():
            for start in range(0, len(samples), args.batch_size):
                batch = samples[start:start + args.batch_size]
                tensors = []
                for row in batch:
                    with Image.open(row["path"]) as image:
                        tensors.append(transforms.clip_basic(image))
                visual = F.normalize(model.encode_image(torch.stack(tensors).to(args.device)).float(), dim=-1)
                e, _ = decomposition(visual)
                y = torch.tensor([row["class_id"] for row in batch], device=args.device)
                # B5A native Z must never select a text using an image label.
                contexts = pc[y] if class_conditioned else pn
                const_contexts = pconst[y, None] if class_conditioned else pconst
                z = F.normalize(tda(e, contexts), dim=-1)
                z_const = F.normalize(tda(e, const_contexts), dim=-1)[:, 0]
                ln_e = F.normalize(tda.layer_norm(e), dim=-1)
                stop = start + len(batch)
                for key, tensor in [("e_unit", F.normalize(e, dim=-1)), ("ln_e_unit", ln_e),
                                    ("z_constant_unit", z_const), (z_key, z)]:
                    arrays[key][start:stop] = tensor.cpu().numpy()
                if class_conditioned:
                    neutral = F.normalize(tda(e, pn), dim=-1)
                    probes["neutral_swap_cosine_shift"][start:stop] = (
                        1 - (z * neutral).sum(-1).clamp(-1, 1)).mean(-1).cpu().numpy()
                    # Post-training probe: hold image/template fixed, replace only the class noun.
                    for c in range(len(class_names)):
                        wrong = F.normalize(tda(e, pc[c]), dim=-1)
                        shifts = (1 - (z * wrong).sum(-1).clamp(-1, 1)).mean(-1)
                        shifts = shifts * (y != c) / (len(class_names) - 1)
                        probes["class_swap_cosine_shift"][start:stop] += shifts.cpu().numpy()
                else:
                    # B5A has no class noun to swap. Adding one is a separate
                    # post-training probe, not a native augmentation result.
                    inserted = F.normalize(tda(e, pc[y]), dim=-1)
                    probes["class_insertion_cosine_shift"][start:stop] = (
                        1 - (z * inserted).sum(-1).clamp(-1, 1)).mean(-1).cpu().numpy()
                print("Extracted {}/{} images; {} native 80-template bank".format(
                    stop, len(samples), template_mode.upper()), flush=True)
    finally:
        for array in arrays.values():
            array.flush()
    np.savez_compressed(out / "intervention_metrics.npz", **probes)
    test_envs = checkpoint.get("test_envs", checkpoint.get("args", {}).get("test_envs"))
    if test_envs is not None and any(not 0 <= i < len(available_domains) for i in test_envs):
        raise ValueError("Checkpoint held-out domain indices do not match this dataset.")
    source_paths = ["analyze_cipt_mechanism.py", "tools/cipt_mechanism_report.py",
                    "domainbed/algorithms/cipt_modules.py", "domainbed/algorithms/cipt_prompt.py",
                    "domainbed/datasets/transforms.py"]
    metadata = {
        "schema_version": 2, "dataset": args.dataset, "class_names": class_names,
        "template_mode": template_mode, "neutral_subject": neutral_subject,
        "z_file": z_key + ".npy", "z_uses_ground_truth_class_word": class_conditioned,
        "native_bank_key": native_bank_key,
        "native_bank_loaded_from_checkpoint": native_bank_key in state,
        "selected_class_ids": class_ids, "domains": domains, "image_count": len(samples),
        "sample_seed": args.sample_seed, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path), "checkpoint_hparams": hparams,
        "checkpoint_args": checkpoint.get("args", {}),
        "checkpoint_test_envs": test_envs,
        "target_domains": None if test_envs is None else [available_domains[i] for i in test_envs],
        "clip_path": str(clip_path.resolve()), "clip_sha256": sha256(clip_path),
        "clip_weights_loaded_from_checkpoint": bool(clip_state), "device": args.device,
        "b5b_bank_loaded_from_checkpoint": "text_features.b5b_text_bank" in state,
        "b5a_bank_loaded_from_checkpoint": "text_features.b5a_text_bank" in state,
        "input_manifest": str(Path(args.sample_manifest).expanduser().resolve()) if args.sample_manifest else None,
        "samples_sha256": sha256(out / "samples.csv"),
        "samples_identity_sha256": hashlib.sha256(json.dumps([
            [row["relative_path"], row["class"], row["domain"]] for row in samples
        ], ensure_ascii=False).encode("utf-8")).hexdigest(),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "source_sha256": {path: sha256(ROOT / path) for path in source_paths},
        "template_count": 80, "class_swap_count": len(class_names) - 1 if class_conditioned else 0,
        "preprocessing": "branch clip_basic; CLIP image feature L2-normalized before causal adapter",
        "coverage": "balanced sample of dataset files; source files can include training images; descriptive analysis only",
        "z_definition": ("L2(TDA(E, B5B bank indexed by ground-truth class)), all 80 templates"
                         if class_conditioned else
                         "L2(TDA(E, saved shared B5A bank)), all 80 templates; labels do not select texts"),
        "ln_e_definition": "L2(trained TDA.layer_norm(E)); zero text-residual control",
        "constant_definition": ("L2(TDA(E, 'a photo of a {true_class}.'))" if class_conditioned else
                                "L2(TDA(E, 'a photo of a {}.'))".format(neutral_subject)),
        "posthoc_probe_definitions": ({
            "class_swap_cosine_shift": "Native class word replaced by every other class, matched template.",
            "neutral_swap_cosine_shift": "Native class word replaced by {}.".format(neutral_subject),
        } if class_conditioned else {
            "class_insertion_cosine_shift": "Native {} replaced by true class word; post-training probe only.".format(neutral_subject),
        }),
    }
    dump_json(out / "metadata.json", json.loads(json.dumps(metadata, default=str)))


def main():
    args = parse_args()
    if not args.report_only:
        extract(args)
    if args.extract_only:
        return
    folder = Path(args.output_dir).expanduser().resolve()
    meta, samples, representations, z, rows, per_image = summarize(folder, args.neighbors)
    plot_metrics(folder, rows, per_image, meta)
    if not args.no_tsne:
        plot_tsne(folder, meta, samples, representations, z, args.tsne_seeds, args.perplexities,
                  args.template_id, args.tsne_iterations, args.pca_dim)
        if not args.skip_template_tsne:
            plot_template_tsne(folder, meta, samples, representations, z, args.tsne_seeds,
                               args.perplexities, args.template_ids, args.tsne_iterations, args.pca_dim)
        if args.overview:
            plot_overview_tsne(folder, meta, samples, representations, z, args.tsne_seeds,
                               args.perplexities, args.template_id, args.tsne_iterations, args.pca_dim)
    import sklearn
    dump_json(folder / "plot_settings.json", {"tsne_seeds": args.tsne_seeds, "perplexities": args.perplexities,
              "template_mode": meta.get("template_mode", meta.get("checkpoint_hparams", {}).get("cipt_template_mode", "b5b")),
              "template_id": args.template_id, "iterations": args.tsne_iterations, "pca_dim": args.pca_dim,
              "template_ids": args.template_ids,
              "template_tsne_generated": not args.no_tsne and not args.skip_template_tsne,
              "overview_generated": not args.no_tsne and args.overview,
              "neighbors": args.neighbors, "sklearn_version": sklearn.__version__, "tsne_generated": not args.no_tsne})
    print("Complete: {}".format(folder / "report.md"))


if __name__ == "__main__":
    main()

