#!/usr/bin/env python3
"""Compare original CIPT E / a fixed TDA context / an E-only RD model.

Place beside train_all.py on feature/cipt-b5b-entropy-mcc-rd. All models use
the original analysis's exact image manifest. No training is performed.
Only load checkpoints you trust. --cache-only redraws this script's cache.
"""
import argparse
import csv
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np

KEYS = ("cipt_e", "cipt_z", "ours_e")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--baseline-dir", type=Path)
    p.add_argument("--cipt-checkpoint", type=Path)
    p.add_argument("--ours-checkpoint", type=Path)
    p.add_argument("--data-dir")
    p.add_argument("--clip-path")
    p.add_argument("--dataset", choices=["PACS", "VLCS"], default="PACS")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--template-id", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--samples-per-class-domain", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--neighbors", type=int, default=5)
    p.add_argument("--tsne-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--perplexities", type=float, nargs="+", default=[10, 30])
    p.add_argument("--tsne-iterations", type=int, default=1000)
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--no-tsne", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    a = p.parse_args()
    if not 0 <= a.template_id < 80 or a.neighbors < 1 or a.batch_size < 1:
        p.error("template-id must be 0..79; neighbors and batch-size must be positive")
    if (a.tsne_iterations < 250 or a.pca_dim < 0 or a.pca_dim == 1
            or any(not np.isfinite(v) or v <= 0 for v in a.perplexities)):
        p.error("Use >=250 iterations, PCA=0 or >=2, and finite positive perplexities")
    if a.samples_per_class_domain < 2:
        p.error("Need >=2 samples per class/domain")
    if not a.cache_only and not all([a.ours_checkpoint, a.data_dir, a.clip_path]):
        p.error("Extraction needs --ours-checkpoint, --data-dir, --clip-path")
    return a


def module_at(name, path):
    if not path.is_file():
        raise FileNotFoundError("Missing {}. Put this script beside train_all.py on the RD branch.".format(path))
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def boolean(x):
    return x.lower() == "true" if isinstance(x, str) else bool(x)


def extract_ours(a, baseline, meta, mechanism):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the training server or --device cpu")
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    ckpt_path = a.ours_checkpoint.expanduser().resolve()
    checkpoint = torch.load(ckpt_path, **options)
    hp = checkpoint.get("model_hparams", checkpoint.get("hparams", {}))
    if not isinstance(hp, dict):
        hp = vars(hp)
    if not boolean(hp.get("cipt_use_de", True)):
        raise ValueError("Ours checkpoint has decomposition disabled")
    if hp.get("cipt_env_inference") != "e_only":
        raise ValueError("Expected an E-only RD checkpoint (cipt_env_inference=e_only)")
    if hp.get("cipt_env_constraint") != "relative_distance" or hp.get("cipt_objective") != "env_anchor":
        raise ValueError("Expected the current env_anchor / relative_distance RD model")
    if checkpoint.get("args", {}).get("dataset", a.dataset) != meta["dataset"]:
        raise ValueError("Checkpoint datasets differ")
    if list(hp.get("cipt_class_names", [])) != meta["class_names"]:
        raise ValueError("Checkpoint class names/order differ; cannot pair the models safely")
    test_envs = checkpoint.get("test_envs", checkpoint.get("args", {}).get("test_envs"))
    if test_envs is None or meta.get("checkpoint_test_envs") is None:
        raise ValueError("Both checkpoints need an explicit held-out-domain record")
    if sorted(test_envs) != sorted(meta["checkpoint_test_envs"]):
        raise ValueError("Models use different held-out domains")
    clip_path = Path(a.clip_path).expanduser().resolve()
    if mechanism.sha256(clip_path) != meta["clip_sha256"]:
        raise ValueError("Local CLIP weights differ from the original CIPT analysis")
    if hp.get("cipt_clip_backbone", "ViT-B/16") != meta["checkpoint_hparams"].get("cipt_clip_backbone", "ViT-B/16"):
        raise ValueError("CLIP backbones differ")
    root = mechanism.resolve_data_root(a.data_dir, a.dataset)
    samples = mechanism.select_samples(root, meta["class_names"], meta["selected_class_ids"],
                                      meta["domains"], a.samples_per_class_domain, a.sample_seed,
                                      baseline / "samples.csv")
    mechanism.validate_samples(samples, meta["selected_class_ids"], meta["domains"])
    random.seed(a.sample_seed)
    np.random.seed(a.sample_seed)
    torch.manual_seed(a.sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.sample_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    state = mechanism.checkpoint_state(checkpoint)
    prompt = mechanism.load_module("_comparison_prompt", a.project_dir / "domainbed/algorithms/cipt_prompt.py")
    modules = mechanism.load_module("_comparison_modules", a.project_dir / "domainbed/algorithms/cipt_modules.py")
    transforms = mechanism.load_module("_comparison_transforms", a.project_dir / "domainbed/datasets/transforms.py")
    model, _ = prompt.load_frozen_clip(hp.get("cipt_clip_backbone", "ViT-B/16"), str(clip_path))
    clip_state = mechanism.substate(state, "clip_model.")
    if clip_state:
        model.load_state_dict(clip_state, strict=True)
    model.to(a.device).eval().requires_grad_(False)
    decomp = modules.CausalDecomposition(int(model.visual.output_dim))
    decomp.load_state_dict(mechanism.substate(state, "causal_decomposition."), strict=True)
    decomp.to(a.device).eval().requires_grad_(False)
    values = []
    with torch.inference_mode():
        for start in range(0, len(samples), a.batch_size):
            tensors = []
            for row in samples[start:start + a.batch_size]:
                with Image.open(row["path"]) as im:
                    tensors.append(transforms.clip_basic(im))
            visual = F.normalize(model.encode_image(torch.stack(tensors).to(a.device)).float(), dim=-1)
            e, _ = decomp(visual)
            values.append(F.normalize(e, dim=-1).cpu().numpy())
            print("Ours E: {}/{} images".format(min(start + a.batch_size, len(samples)), len(samples)), flush=True)
    ours_meta = {"checkpoint": str(ckpt_path), "checkpoint_sha256": mechanism.sha256(ckpt_path),
                "checkpoint_hparams": hp, "checkpoint_test_envs": test_envs,
                "clip_weights_restored": bool(clip_state)}
    return np.concatenate(values), samples, ours_meta


def extract(a, report):
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise ValueError("Use a NEW --output-dir, or --cache-only to redraw completed extraction")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.project_dir))
    mechanism = module_at("_comparison_mechanism", a.project_dir / "analyze_cipt_mechanism.py")
    baseline = a.baseline_dir.expanduser().resolve() if a.baseline_dir else None
    required = ("metadata.json", "samples.csv", "e_unit.npy", "z_class_unit.npy")
    if baseline is None or not all((baseline / x).is_file() for x in required):
        if a.cipt_checkpoint is None:
            raise ValueError("Baseline arrays unavailable: pass --cipt-checkpoint to re-extract")
        manifest = baseline / "samples.csv" if baseline and (baseline / "samples.csv").is_file() else None
        baseline = a.output_dir / "cipt_extraction"
        original_args = mechanism.parse_args([
            "--dataset", a.dataset, "--checkpoint", str(a.cipt_checkpoint),
            "--data-dir", a.data_dir, "--clip-path", a.clip_path,
            "--output-dir", str(baseline), "--sample-seed", str(a.sample_seed),
            "--samples-per-class-domain", str(a.samples_per_class_domain),
            "--batch-size", str(a.batch_size), "--device", a.device,
            *(["--sample-manifest", str(manifest)] if manifest else []),
        ])
        mechanism.extract(original_args)
    meta = json.loads((baseline / "metadata.json").read_text(encoding="utf-8"))
    mechanism.validate_original_checkpoint(meta["checkpoint_hparams"])
    if meta["dataset"] != a.dataset:
        raise ValueError("Baseline dataset differs from --dataset")
    if a.cipt_checkpoint and mechanism.sha256(a.cipt_checkpoint.expanduser().resolve()) != meta["checkpoint_sha256"]:
        raise ValueError("Baseline cache was extracted from a different CIPT checkpoint")
    e = np.load(baseline / "e_unit.npy", allow_pickle=False)
    z_bank = np.load(baseline / "z_class_unit.npy", mmap_mode="r", allow_pickle=False)
    ours, samples, ours_meta = extract_ours(a, baseline, meta, mechanism)
    z = np.asarray(z_bank[:, a.template_id, :]).copy()
    if e.shape != ours.shape or z.shape != e.shape or len(samples) != len(e):
        raise ValueError("Paired feature shapes differ")
    features = {"cipt_e": report.unit(e), "cipt_z": report.unit(z), "ours_e": report.unit(ours),
                "classes": np.asarray([r["class_id"] for r in samples]),
                "domains": np.asarray([r["domain"] for r in samples]),
                "relative_paths": np.asarray([r["relative_path"] for r in samples])}
    provenance = {"baseline_metadata": meta, "ours_metadata": ours_meta,
                  "template_id": a.template_id, "neighbors": a.neighbors,
                  "baseline_directory": str(baseline),
                  "comparison": "E vs fixed-template TDA Z vs E-only RD; L2-normalized",
                  "z_caution": "Original Z uses ground-truth-class text: diagnostic only, not label-free inference or accuracy.",
                  "coverage": meta.get("coverage"),
                  "paired_manifest_sha256": mechanism.sha256(baseline / "samples.csv")}
    shutil.copyfile(baseline / "samples.csv", a.output_dir / "paired_samples.csv")
    report.dump_json(a.output_dir / "provenance.json", json.loads(json.dumps(provenance, default=str)))
    np.savez_compressed(a.output_dir / "paired_features.npz", **features)
    return features, provenance


def measure(features, meta, a, report):
    classes, domains = features["classes"], features["domains"]
    names = meta["baseline_metadata"]["class_names"]
    per_class, summary = [], []
    for key in KEYS:
        x = features[key]
        rows = []
        for c in np.unique(classes):
            mask = classes == c
            row = report.domain_metrics(x[mask], domains[mask], a.neighbors)
            row.update(representation=key, class_name=names[int(c)])
            row["cross_domain_neighbor_fraction"] = 1 - row["same_domain_neighbor_fraction"]
            rows.append(row)
        per_class.extend(rows)
        distance = report.cosine_distance_matrix(x)
        weights = report.neighbor_weights(distance, a.neighbors)
        label_match = classes[:, None] == classes[None, :]
        class_purity = float(((weights * label_match).sum(1) / weights.sum(1)).mean())
        upper = np.triu(np.ones_like(label_match), 1)
        cross_domain = domains[:, None] != domains[None, :]
        within_cross = float(distance[upper & label_match & cross_domain].mean())
        between_class = float(distance[upper & ~label_match].mean())
        avg = {k: float(np.mean([r[k] for r in rows])) for k in
               ["cross_domain_neighbor_fraction", "domain_silhouette", "neighbor_excess", "domain_cosine_gap"]}
        avg.update(representation=key, class_neighbor_purity=class_purity,
                   same_class_cross_domain_distance=within_cross,
                   different_class_distance=between_class,
                   distance_ratio=within_cross / between_class if between_class > 1e-12 else None)
        summary.append(avg)
    report.write_csv(a.output_dir / "metrics_per_class.csv", per_class)
    report.write_csv(a.output_dir / "metrics_summary.csv", summary)
    return summary


def figures(features, provenance, summary, a, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    names = provenance["baseline_metadata"]["class_names"]
    titles = ["CIPT E", "CIPT Z (template {})".format(provenance["template_id"]), "Ours E"]
    palette = plt.get_cmap("tab10")
    classes, domains = features["classes"], features["domains"]
    domain_names = sorted(set(domains.tolist()))
    class_ids = sorted(set(classes.tolist()))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, metric, title in zip(axes,
            ["cross_domain_neighbor_fraction", "class_neighbor_purity", "distance_ratio"],
            ["Within-class cross-domain neighbors", "Class neighbor purity", "Same-class cross-domain / different-class distance"]):
        values = [row[metric] for row in summary]
        ax.bar(range(3), [np.nan if v is None else v for v in values],
               color=["#9BAAC4", "#B3C4C3", "#D99172"])
        ax.set_xticks(range(3), titles, rotation=12, fontsize=8)
        ax.set_title(title, fontsize=9)
        if metric != "distance_ratio":
            ax.set_ylim(0, 1.05)
        for i, value in enumerate(values):
            if value is not None:
                ax.text(i, value, "{:.3f}".format(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(a.output_dir / ("comparison_metrics." + ext), dpi=220)
    plt.close(fig)
    if a.no_tsne:
        return
    folder = a.output_dir / "tsne"
    folder.mkdir(exist_ok=True)
    targets = provenance["baseline_metadata"].get("target_domains") or []
    for group in [None, *class_ids]:
        mask = np.ones(len(classes), bool) if group is None else classes == group
        arrays = [features[k][mask] for k in KEYS]
        group_name = "all_classes" if group is None else "class_{}".format(names[group])
        for seed in a.tsne_seeds:
            for perplexity in a.perplexities:
                print("t-SNE {} seed={} perplexity={}".format(group_name, seed, perplexity), flush=True)
                coordinates = report.joint_tsne(arrays, seed, perplexity, a.tsne_iterations, a.pca_dim)
                fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True, sharey=True)
                for ax, xy, title in zip(axes, coordinates, titles):
                    for di, domain in enumerate(domain_names):
                        for ci, c in enumerate(class_ids):
                            chosen = (domains[mask] == domain) & (classes[mask] == c)
                            if chosen.any():
                                ax.scatter(xy[chosen, 0], xy[chosen, 1], s=17, alpha=.75,
                                           color=palette(ci if group is None else di),
                                           marker=["o", "s", "^", "D"][di % 4] if group is None else "o",
                                           linewidths=0)
                    ax.set_title(title)
                    ax.set_xticks([])
                    ax.set_yticks([])
                handles = []
                if group is None:
                    handles += [Line2D([], [], color=palette(ci), marker="o", ls="", label=names[c])
                                for ci, c in enumerate(class_ids)]
                handles += [Line2D([], [], color="#555555" if group is None else palette(di), ls="",
                                   marker=["o", "s", "^", "D"][di % 4] if group is None else "o",
                                   label=domain + (" [target]" if domain in targets else ""))
                            for di, domain in enumerate(domain_names)]
                fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False, fontsize=8)
                fig.suptitle("{} | identical images | shared PCA/t-SNE | seed {} | perplexity {}".format(
                    group_name, seed, perplexity), fontsize=10)
                fig.tight_layout(rect=(0, .15, 1, .94))
                stem = "{}_seed{}_p{:g}".format(group_name, seed, perplexity)
                for ext in ["png", "pdf"]:
                    fig.savefig(folder / (stem + "." + ext), dpi=220)
                plt.close(fig)
                np.savez_compressed(folder / (stem + ".npz"),
                                    **dict(zip(KEYS, coordinates)), classes=classes[mask], domains=domains[mask],
                                    relative_paths=features["relative_paths"][mask])


def main():
    a = arguments()
    a.project_dir = a.project_dir.expanduser().resolve()
    a.output_dir = a.output_dir.expanduser().resolve()
    report = module_at("_comparison_report", a.project_dir / "tools/cipt_mechanism_report.py")
    if a.cache_only:
        with np.load(a.output_dir / "paired_features.npz", allow_pickle=False) as archive:
            features = {k: archive[k] for k in [*KEYS, "classes", "domains", "relative_paths"]}
        provenance = json.loads((a.output_dir / "provenance.json").read_text(encoding="utf-8"))
    else:
        features, provenance = extract(a, report)
    if len(np.unique(features["classes"])) < 2:
        raise ValueError("Use >=2 classes to check class separation alongside domain mixing")
    for key in KEYS:
        if not np.isfinite(features[key]).all() or (np.linalg.norm(features[key], axis=1) < 1e-9).any():
            raise ValueError("Invalid or zero feature vectors: " + key)
    summary = measure(features, provenance, a, report)
    figures(features, provenance, summary, a, report)
    report.dump_json(a.output_dir / "comparison_settings.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    lines = ["# CIPT and E-only RD comparison", "",
             "Same image manifest, class order, held-out domains and CLIP checkpoint. High-dimensional metrics use unit features.", "",
             "| Representation | Cross-domain neighbors within class | Class neighbor purity | Distance ratio |",
             "|---|---:|---:|---:|"]
    for row in summary:
        ratio = "undefined" if row["distance_ratio"] is None else "{:.4f}".format(row["distance_ratio"])
        lines.append("| {} | {:.4f} | {:.4f} | {} |".format(row["representation"], row["cross_domain_neighbor_fraction"], row["class_neighbor_purity"], ratio))
    lines += ["", "Interpretation:",
              "- More cross-domain same-class neighbors is supportive only if class purity is retained; check target-domain accuracy separately.",
              "- This is descriptive analysis, including source images that may have been used for training. It does not establish causal identification.",
              "- Original CIPT Z uses ground-truth class-conditioned text and one predeclared template, not a label-free predictor or the complete inference procedure.",
              "- The E-only model has no TDA representation here. Text-swap invariance of its E is not evidence of a learned improvement.",
              "- Each figure shares one joint PCA/t-SNE fit; do not quantify inter-model displacement or 2D cluster distances.",
              "- These comparisons do not isolate NR and RD contributions. For attribution, repeat the same analysis with matched E/S, +NR, +RD and +NR+RD checkpoints.",
              "- The repository's TE*_5000.pth files store ordinary models; their filenames do not imply SWAD averaging. Compare checkpoint selection rules explicitly.", ""]
    (a.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Complete: {}".format(a.output_dir), flush=True)


if __name__ == "__main__":
    main()
