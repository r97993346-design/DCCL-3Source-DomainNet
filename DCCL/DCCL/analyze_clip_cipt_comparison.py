#!/usr/bin/env python3
"""Add frozen CLIP V to a paired, trained CIPT/B5A/RD+NR E comparison.

Reuses existing E arrays, restores the baseline checkpoint's frozen CLIP, and
checks reconstructed baseline E before saving V. No training is performed.
All metrics use original-dimensional unit features; t-SNE is illustrative.
"""
import argparse
import csv
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
from analyze_cipt_text_features import checkpoint_state, load_module, resolve_data_root, select_samples, substate
from analyze_cipt_mechanism import sha256, validate_samples
from cipt_mechanism_report import (cosine_distance_matrix, domain_metrics, dump_json,
                                   joint_tsne, neighbor_weights, unit, write_csv)

KEYS = ["clip_v", "cipt_e", "b5a_e", "ours_e"]
TITLES = ["Frozen CLIP V", "Original CIPT E", "Class-agnostic B5A E", "RD + NR E"]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def identity(rows):
    return [(r["relative_path"], int(r["class_id"]), r["class"], r["domain"]) for r in rows]


def load_inputs(comparison_dir, b5a_dir):
    cmp, b5a = Path(comparison_dir), Path(b5a_dir)
    prov = json.loads((cmp / "provenance.json").read_text())
    meta = prov["baseline_metadata"]
    other = json.loads((b5a / "metadata.json").read_text())
    samples, b_samples = read_csv(cmp / "paired_samples.csv"), read_csv(b5a / "samples.csv")
    if identity(samples) != identity(b_samples):
        raise ValueError("B5A and comparison manifests differ in images, labels or order.")
    if other.get("template_mode") != "b5a" or other.get("z_uses_ground_truth_class_word") is not False:
        raise ValueError("Choose the separately trained shared-bank B5A extraction.")
    if other["class_names"] != meta["class_names"] or other["clip_sha256"] != meta["clip_sha256"]:
        raise ValueError("B5A class mapping or CLIP file differs from the baseline.")
    with np.load(cmp / "paired_features.npz", allow_pickle=False) as archive:
        pack = {k: archive[k] for k in archive.files}
    expected = {
        "relative_paths": [r["relative_path"] for r in samples],
        "classes": [int(r["class_id"]) for r in samples],
        "domains": [r["domain"] for r in samples],
    }
    for k, value in expected.items():
        if pack[k].tolist() != value:
            raise ValueError("Comparison feature/manifest mismatch: " + k)
    pack = {k: pack[k] for k in ["cipt_e", "ours_e", "relative_paths", "classes", "domains"]}
    pack["b5a_e"] = np.load(b5a / "e_unit.npy", allow_pickle=False)
    for k in ["cipt_e", "b5a_e", "ours_e"]:
        if pack[k].shape != pack["cipt_e"].shape or len(pack[k]) != len(samples):
            raise ValueError("Feature shape mismatch: " + k)
        unit(pack[k])  # Reject nonfinite or zero features.
    if len(set(pack["relative_paths"].tolist())) != len(samples):
        raise ValueError("Duplicate images in paired archive.")
    return pack, samples, meta, prov, other


def extract(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn.functional as F
    from PIL import Image

    pack, samples, baseline, prov, b5a = load_inputs(args.comparison_dir, args.b5a_dir)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be empty; use a new --output-dir or --report-only.")
    checkpoint_path, clip_path = Path(args.checkpoint).expanduser(), Path(args.clip_path).expanduser()
    if sha256(checkpoint_path) != baseline["checkpoint_sha256"]:
        raise ValueError("--checkpoint must be the original CIPT checkpoint used by the comparison.")
    if sha256(clip_path) != baseline["clip_sha256"]:
        raise ValueError("--clip-path hash differs from the original extraction.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run on the GPU server or pass --device cpu.")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False  # User's own local trained checkpoint.
    checkpoint = torch.load(checkpoint_path, **kwargs)
    state = checkpoint_state(checkpoint)
    hp = checkpoint.get("model_hparams", checkpoint.get("hparams", {}))
    prompt = load_module("_clip_comparison_prompt", ROOT / "domainbed/algorithms/cipt_prompt.py")
    modules = load_module("_clip_comparison_modules", ROOT / "domainbed/algorithms/cipt_modules.py")
    transforms = load_module("_clip_comparison_transforms", ROOT / "domainbed/datasets/transforms.py")
    model, _ = prompt.load_frozen_clip(hp.get("cipt_clip_backbone", "ViT-B/16"), str(clip_path))
    saved_clip = substate(state, "clip_model.")
    if saved_clip:
        model.load_state_dict(saved_clip, strict=True)
    model.to(args.device).eval().requires_grad_(False)
    decomposition = modules.CausalDecomposition(int(model.visual.output_dim))
    decomposition.load_state_dict(substate(state, "causal_decomposition."), strict=True)
    decomposition.to(args.device).eval().requires_grad_(False)
    names, domains = baseline["class_names"], baseline["domains"]
    replay = select_samples(resolve_data_root(args.data_dir, baseline["dataset"]), names,
                            sorted(set(pack["classes"].tolist())), domains, 25, 0,
                            Path(args.comparison_dir) / "paired_samples.csv")
    validate_samples(replay, sorted(set(pack["classes"].tolist())), domains)
    if identity(samples) != identity(replay):
        raise ValueError("Replayed manifest differs from the feature archive.")
    visual, reconstructed = [], []
    with torch.inference_mode():
        for start in range(0, len(replay), args.batch_size):
            tensors = []
            for row in replay[start:start + args.batch_size]:
                with Image.open(row["path"]) as image:
                    tensors.append(transforms.clip_basic(image))
            v = F.normalize(model.encode_image(torch.stack(tensors).to(args.device)).float(), dim=-1)
            e, _ = decomposition(v)
            visual.append(v.cpu().numpy())
            reconstructed.append(F.normalize(e, dim=-1).cpu().numpy())
            print("Extracted frozen CLIP V: {}/{}".format(min(start + args.batch_size, len(replay)), len(replay)), flush=True)
    restored_e = np.concatenate(reconstructed)
    # Catch a different checkpoint, preprocessing or input mapping before producing a comparison.
    np.testing.assert_allclose(restored_e, pack["cipt_e"], atol=2e-4, rtol=2e-4,
                               err_msg="Reconstructed CIPT E differs from the archived baseline.")
    pack["clip_v"] = np.concatenate(visual)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "v_unit.npy", pack["clip_v"])
    np.savez_compressed(out / "paired_features.npz", **pack)
    write_csv(out / "samples.csv", replay)
    dump_json(out / "metadata.json", {
        "dataset": baseline["dataset"], "class_names": names, "domains": domains,
        "target_domains": baseline.get("target_domains"), "image_count": len(replay),
        "feature_dimension": pack["clip_v"].shape[1], "representations": KEYS,
        "clip_sha256": baseline["clip_sha256"], "checkpoint_sha256": baseline["checkpoint_sha256"],
        "clip_weights_restored_from_checkpoint": bool(saved_clip),
        "baseline_e_max_abs_reconstruction_error": float(np.max(abs(restored_e - pack["cipt_e"]))),
        "baseline_metadata": baseline, "b5a_metadata": b5a, "ours_metadata": prov["ours_metadata"],
        "source_files_sha256": {str(p): sha256(p) for p in [
            ROOT / "analyze_clip_cipt_comparison.py", ROOT / "tools/cipt_mechanism_report.py",
            ROOT / "domainbed/datasets/transforms.py"]},
        "source_archives_sha256": {str(p): sha256(p) for p in [
            Path(args.comparison_dir) / "paired_features.npz", Path(args.b5a_dir) / "e_unit.npy"]},
        "preprocessing": "branch clip_basic; L2 normalized encode_image before decomposition",
        "coverage": "Paired descriptive analysis; source images may include training samples; no accuracy evaluation.",
    })


def report(folder, neighbors):
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text())
    with np.load(folder / "paired_features.npz", allow_pickle=False) as archive:
        pack = {k: archive[k] for k in archive.files}
    y, domains = pack["classes"], pack["domains"]
    same_class = y[:, None] == y[None, :]
    cross = domains[:, None] != domains[None, :]
    upper = np.triu(np.ones(same_class.shape, dtype=bool), 1)
    rows, summary = [], []
    for key in KEYS:
        features = pack[key]
        local = []
        for c in sorted(set(y.tolist())):
            mask = y == c
            metrics = domain_metrics(features[mask], domains[mask], neighbors)
            metrics["cross_domain_neighbor_fraction"] = 1 - metrics["same_domain_neighbor_fraction"]
            row = {"class_id": c, "class_name": meta["class_names"][c], "representation": key, **metrics}
            rows.append(row); local.append(metrics)
        d = cosine_distance_matrix(features)
        w = neighbor_weights(d, neighbors)
        intra = float(d[upper & same_class & cross].mean())
        inter = float(d[upper & ~same_class].mean())
        summary.append({"representation": key, **{
            k: float(np.mean([m[k] for m in local])) for k in [
                "cross_domain_neighbor_fraction", "domain_silhouette", "neighbor_excess", "domain_cosine_gap"]},
            "class_neighbor_purity": float(((w * same_class).sum(1) / w.sum(1)).mean()),
            "same_class_cross_domain_distance": intra, "different_class_distance": inter,
            "distance_ratio": intra / inter})
    write_csv(folder / "metrics_per_class.csv", rows)
    write_csv(folder / "metrics_summary.csv", summary)
    dump_json(folder / "summary.json", {"metadata": meta, "neighbors": neighbors, "metrics": summary})
    lines = ["# Frozen CLIP versus trained E", "",
             "| Representation | Same-class cross-domain {}NN | Domain silhouette | Class-neighbor purity |".format(neighbors),
             "|---|---:|---:|---:|"]
    lines += ["| {representation} | {cross_domain_neighbor_fraction:.4%} | {domain_silhouette:.4f} | {class_neighbor_purity:.4%} |".format(**m) for m in summary]
    lines += ["", "All metrics use original-dimensional unit features. t-SNE is illustrative.",
              "Source images can include training samples. Neighbor purity is not classification accuracy.",
              "Comparing trained E with frozen V describes the entire training path, not an isolated decomposition ablation.",
              "Current RD+NR uses class-related text anchors and different loss settings; no individual module attribution."]
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pack, meta


def plots(folder, pack, meta, seeds, perplexities, iterations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder = Path(folder) / "tsne"; folder.mkdir(exist_ok=True)
    y, domains = pack["classes"], pack["domains"]
    domain_names = list(meta["domains"])
    colors = plt.get_cmap("tab10"); markers = ["o", "s", "^", "D", "P", "X"]
    for seed in seeds:
        for p in perplexities:
            for c in [None] + sorted(set(y.tolist())):
                mask = np.ones(len(y), bool) if c is None else y == c
                coords = joint_tsne([pack[k][mask] for k in KEYS], seed, p, iterations, 50)
                fig, axs = plt.subplots(1, 4, figsize=(17, 4.7), sharex=True, sharey=True)
                for ax, xy, title in zip(axs, coords, TITLES):
                    for j, domain in enumerate(domain_names):
                        if c is None:
                            for label in sorted(set(y.tolist())):
                                m = (domains[mask] == domain) & (y[mask] == label)
                                ax.scatter(*xy[m].T, s=11, color=colors(label), marker=markers[j % len(markers)], alpha=.8)
                        else:
                            m = domains[mask] == domain
                            ax.scatter(*xy[m].T, s=15, color=colors(j), alpha=.8)
                    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
                if c is None:
                    class_legend = [Line2D([], [], marker="o", linestyle="", color=colors(i), label=n)
                                    for i, n in enumerate(meta["class_names"])]
                    fig.legend(handles=class_legend, loc="lower center", bbox_to_anchor=(.5, .055), ncol=len(class_legend), fontsize=8)
                    handles = [Line2D([], [], marker=markers[j % len(markers)], linestyle="", color="gray", label=n)
                               for j, n in enumerate(domain_names)]
                else:
                    handles = [Line2D([], [], marker="o", linestyle="", color=colors(j), label=n)
                               for j, n in enumerate(domain_names)]
                fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
                name = "all_classes" if c is None else "class_" + meta["class_names"][c]
                title = "All classes: color=class, marker=domain" if c is None else meta["class_names"][c] + ": color=domain"
                fig.suptitle(title + "; one joint PCA/t-SNE fit; seed={}, p={:g}".format(seed, p))
                fig.tight_layout(rect=(0, .13 if c is None else .07, 1, .94))
                stem = "{}_seed{}_p{:g}".format(name, seed, p)
                for suffix in [".png", ".pdf"]:
                    fig.savefig(folder / (stem + suffix), dpi=160, bbox_inches="tight")
                plt.close(fig)
                np.savez_compressed(folder / (stem + ".npz"), **dict(zip(KEYS, coords)),
                                    classes=y[mask], domains=domains[mask], relative_paths=pack["relative_paths"][mask])
                print("Rendered " + stem, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint"); p.add_argument("--clip-path"); p.add_argument("--data-dir")
    p.add_argument("--comparison-dir"); p.add_argument("--b5a-dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=8); p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--neighbors", type=int, default=5)
    p.add_argument("--tsne-seeds", nargs="+", type=int, default=[0])
    p.add_argument("--perplexities", nargs="+", type=float, default=[30])
    p.add_argument("--tsne-iterations", type=int, default=1000)
    p.add_argument("--report-only", action="store_true"); p.add_argument("--no-tsne", action="store_true")
    args = p.parse_args(argv)
    if not args.report_only and any(not getattr(args, k) for k in ["checkpoint", "clip_path", "data_dir", "comparison_dir", "b5a_dir"]):
        p.error("Extraction needs checkpoint, clip-path, data-dir, comparison-dir and b5a-dir.")
    if min(args.batch_size, args.cpu_threads, args.neighbors) < 1 or args.tsne_iterations < 250:
        p.error("Use positive sizes/threads/neighbors and >=250 t-SNE iterations.")
    if any(not np.isfinite(x) or x <= 0 for x in args.perplexities):
        p.error("Perplexities must be finite and positive.")
    return args


def main():
    args = parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.cpu_threads):
        if not args.report_only:
            extract(args)
        pack, meta = report(args.output_dir, args.neighbors)
        if not args.no_tsne:
            plots(args.output_dir, pack, meta, args.tsne_seeds, args.perplexities, args.tsne_iterations)
        dump_json(Path(args.output_dir) / "plot_settings.json", {
            "neighbors": args.neighbors, "tsne_seeds": args.tsne_seeds, "perplexities": args.perplexities,
            "iterations": args.tsne_iterations, "pca_dim": 50, "joint_representations": KEYS,
            "tsne_generated": not args.no_tsne})
    print("Complete: " + str(Path(args.output_dir) / "report.md"), flush=True)


if __name__ == "__main__":
    main()
