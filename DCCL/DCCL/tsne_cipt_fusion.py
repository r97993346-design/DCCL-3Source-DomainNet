#!/usr/bin/env python3
"""CIPT inference visualization: every image uses every candidate text class.

Place beside train_all.py, or pass --repo-root.
For each selected image:
  class text -> [all checkpoint classes, 80 templates, embedding dimension]
  empty slot -> [80, embedding dimension]
  subject    -> [80, embedding dimension]

Stars are trained classification text anchors restored from the checkpoint.
Fusion does not use image ground-truth labels. Labels only choose input images
and identify their correct anchor in plots. No point-cloud means are used as centers.
All fused features and classification anchors share one joint t-SNE embedding.
"""

import argparse
import csv
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE

GROUPS = ("class", "empty", "subject")
GROUP_TITLES = ("All candidate-class texts", "Empty class slot", "Subject text")
MARKERS = ("o", "x", "^")
PACS_CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trained CIPTDCCL TE*.pth checkpoint")
    parser.add_argument("--clip-path", required=True, help="Local ViT-B-16.pt used for training")
    parser.add_argument("--data-dir", help="PACS directory, or its parent")
    parser.add_argument("--domain", default="photo", help="Select one image per class from this domain")
    parser.add_argument("--classes", nargs="+", default=list(PACS_CLASSES),
                        help="Image classes to sample; text candidates always include all checkpoint classes")
    parser.add_argument("--image", action="append", default=[], metavar="CLASS=PATH",
                        help="Specify an image for a class; repeat to fix every image manually")
    parser.add_argument("--repo-root", help="Directory containing train_all.py and domainbed/")
    parser.add_argument("--output-dir", default="train_output/ablation/fusion_tsne_all_candidates")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30)
    args = parser.parse_args()
    if not args.classes or len(set(args.classes)) != len(args.classes):
        parser.error("--classes must contain distinct class names")
    return args


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_repo_root(explicit):
    candidates = [Path(explicit)] if explicit else [
        Path(__file__).resolve().parent,
        Path.cwd(), Path.cwd() / "DCCL/DCCL", Path.cwd() / "DCCL",
    ]
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "domainbed/algorithms/cipt_modules.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Place this script beside train_all.py, or pass --repo-root to that directory."
    )


def checkpoint_state(checkpoint):
    state = checkpoint.get("model_dict", checkpoint.get("state_dict", checkpoint))
    suffix = "causal_decomposition.causal_adapter.weight"
    matches = [key for key in state if isinstance(key, str) and key.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError("Expected one trained CIPT causal adapter; found {}".format(matches))
    prefix = matches[0][:-len(suffix)]
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def substate(state, prefix):
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def canonical(name):
    return str(name).replace("_", " ").strip().lower()


def select_images(args):
    manual = {}
    classes = {canonical(name): name for name in args.classes}
    for entry in args.image:
        if "=" not in entry:
            raise ValueError("--image must have the form dog=/absolute/path/image.jpg")
        name, filename = entry.split("=", 1)
        if canonical(name) not in classes:
            raise ValueError("Image class {} is not in --classes".format(name))
        name = classes[canonical(name)]
        if name in manual:
            raise ValueError("Specify only one image for {}".format(name))
        path = Path(filename).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        manual[name] = path

    root = None
    if any(name not in manual for name in args.classes):
        if not args.data_dir:
            raise ValueError("Pass --data-dir, or one --image CLASS=PATH for every class")
        base = Path(args.data_dir).expanduser().resolve()
        for candidate in (base / "PACS", base, base / "PACS/kfold", base / "kfold"):
            if (candidate / args.domain).is_dir():
                root = candidate
                break
        if root is None:
            raise FileNotFoundError("Cannot find domain {} under {}".format(args.domain, base))

    rng = np.random.default_rng(args.seed)
    rows = []
    for name in args.classes:
        if name in manual:
            path, domain = manual[name], "manual"
        else:
            folders = {canonical(p.name): p for p in (root / args.domain).iterdir() if p.is_dir()}
            if canonical(name) not in folders:
                raise FileNotFoundError("Class folder {} missing in {}".format(name, root / args.domain))
            candidates = sorted(
                p.resolve() for p in folders[canonical(name)].rglob("*")
                if p.is_file() and p.suffix.lower() in EXTENSIONS
            )
            if not candidates:
                raise FileNotFoundError("No images for {}".format(name))
            path, domain = candidates[int(rng.integers(len(candidates)))], args.domain
        rows.append({"class": name, "domain": domain, "path": str(path)})
    return rows


def unit(array):
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)


def project(features, perplexity, seed):
    if not np.isfinite(features).all():
        raise ValueError("Non-finite fused features")
    if not np.isfinite(perplexity) or not 0 < perplexity < len(features):
        raise ValueError("perplexity must be finite, positive, and smaller than the point count")
    kwargs = dict(n_components=2, metric="cosine", init="pca",
                  learning_rate=200.0, perplexity=perplexity, random_state=seed)
    key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    kwargs[key] = 1500
    return TSNE(**kwargs).fit_transform(features)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def style(ax, title, xy):
    span = np.maximum(np.ptp(xy, axis=0), 1.0)
    low, high = xy.min(axis=0) - span * 0.07, xy.max(axis=0) + span * 0.07
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def mark_center(ax, center, color, label=None):
    ax.scatter(center[0], center[1], marker="*", s=200, color=color,
               edgecolors="black", linewidths=0.7, zorder=5)
    if label:
        ax.annotate(label, center, xytext=(5, 6), textcoords="offset points",
                    fontsize=8, zorder=6)


def flatten_features(raw_z, anchors):
    """The class condition is a full image x candidate class x template product."""
    parts = [unit(raw_z[group]).reshape(-1, anchors.shape[-1]) for group in GROUPS]
    counts = [len(part) for part in parts]
    return np.concatenate(parts + [unit(anchors)], axis=0), counts


def split_coordinates(flat_xy, raw_z, counts, num_anchors):
    xyz = {}
    start = 0
    for group, count in zip(GROUPS, counts):
        xyz[group] = flat_xy[start:start + count].reshape(*raw_z[group].shape[:-1], 2)
        start += count
    centers = flat_xy[start:start + num_anchors]
    if start + num_anchors != len(flat_xy):
        raise ValueError("Unexpected coordinate count")
    return xyz, centers


def plot_results(xyz, centers, image_classes, candidate_classes, out):
    palette = plt.get_cmap("tab10")
    candidate_colors = [palette(i % 10) for i in range(len(candidate_classes))]
    image_colors = [palette(i % 10) for i in range(len(image_classes))]
    class_lookup = {canonical(name): i for i, name in enumerate(candidate_classes)}
    anchor_indices = [class_lookup[canonical(name)] for name in image_classes]
    all_xy = np.concatenate(
        [xyz[group].reshape(-1, 2) for group in GROUPS] + [centers], axis=0
    )

    # Overview: colors identify the input image class, not the candidate text class.
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for group_index, (group, ax) in enumerate(zip(GROUPS, axes)):
        for image_index, image_class in enumerate(image_classes):
            points = xyz[group][image_index].reshape(-1, 2)
            ax.scatter(points[:, 0], points[:, 1], s=9, alpha=0.4,
                       color=image_colors[image_index], label=image_class)
            mark_center(ax, centers[anchor_indices[image_index]],
                        image_colors[image_index], image_class)
        style(ax, GROUP_TITLES[group_index], all_xy)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Colors: input image class; stars: trained classification text anchors", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "tsne_three_groups.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 9))
    for group_index, group in enumerate(GROUPS):
        for image_index in range(len(image_classes)):
            points = xyz[group][image_index].reshape(-1, 2)
            ax.scatter(points[:, 0], points[:, 1], marker=MARKERS[group_index],
                       s=8, alpha=0.3, color=image_colors[image_index])
    for image_index, image_class in enumerate(image_classes):
        mark_center(ax, centers[anchor_indices[image_index]], image_colors[image_index], image_class)
    handles = [
        Line2D([], [], color=image_colors[i], marker="o", linestyle="none",
               label="Image: " + name)
        for i, name in enumerate(image_classes)
    ]
    handles += [
        Line2D([], [], color="gray", marker=MARKERS[i], linestyle="none",
               label=GROUP_TITLES[i])
        for i in range(3)
    ]
    handles += [Line2D([], [], color="gray", marker="*", markersize=13,
                       linestyle="none", label="Correct classification text anchor")]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left",
              bbox_to_anchor=(1.01, 1))
    style(ax, "All images: all candidate classes / empty / subject", all_xy)
    fig.tight_layout()
    fig.savefig(out / "tsne_joint.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Each image gets a comparison with all candidate classes and the SAME true anchor.
    per_class = out / "per_class"
    per_class.mkdir(exist_ok=True)
    for image_index, image_class in enumerate(image_classes):
        true_anchor = centers[anchor_indices[image_index]]
        image_xy = np.concatenate(
            [xyz[group][image_index].reshape(-1, 2) for group in GROUPS]
            + [true_anchor[None, :]], axis=0
        )
        fig, axes = plt.subplots(1, 3, figsize=(19, 6))
        for candidate_index, candidate_class in enumerate(candidate_classes):
            points = xyz["class"][image_index, candidate_index]
            axes[0].scatter(
                points[:, 0], points[:, 1], s=19, alpha=0.7,
                color=candidate_colors[candidate_index], label="Text: " + candidate_class
            )
        axes[0].legend(frameon=False, fontsize=8, ncol=2)
        for ax, group, color in [
            (axes[1], "empty", "#64717d"), (axes[2], "subject", "#9864bd")
        ]:
            points = xyz[group][image_index]
            ax.scatter(points[:, 0], points[:, 1], s=21, alpha=0.75, color=color)
        for group_index, ax in enumerate(axes):
            mark_center(ax, true_anchor, "#e5b832", "Correct class: " + image_class)
            title = GROUP_TITLES[group_index]
            if group_index == 0:
                title += " ({} x 80 points)".format(len(candidate_classes))
            else:
                title += " (80 points)"
            style(ax, title, image_xy)
        fig.suptitle("One {} image; common reference: its trained {} classification anchor".format(
            image_class, image_class), fontsize=14)
        fig.tight_layout()
        safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in image_class)
        fig.savefig(per_class / (safe_name + ".png"), dpi=300)
        plt.close(fig)


def save_results(out, raw_z, xyz, anchors, center_xy, image_classes, candidate_classes,
                 samples, texts, visual, causal, metadata):
    out.mkdir(parents=True, exist_ok=True)
    plot_results(xyz, center_xy, image_classes, candidate_classes, out)
    write_csv(out / "selected_images.csv", ["class", "domain", "path"], samples)

    centers_rows = [
        {
            "class": name, "center_type": "trained_classification_text_anchor",
            "center_x": float(center_xy[i, 0]), "center_y": float(center_xy[i, 1]),
        }
        for i, name in enumerate(candidate_classes)
    ]
    write_csv(out / "centers.csv",
              ["class", "center_type", "center_x", "center_y"], centers_rows)
    point_rows = []
    for image_index, image_class in enumerate(image_classes):
        for candidate_index, candidate_class in enumerate(candidate_classes):
            for template_index, text in enumerate(texts["class"][candidate_index]):
                coordinate = xyz["class"][image_index, candidate_index, template_index]
                point_rows.append({
                    "condition": "class", "image_class": image_class,
                    "text_class": candidate_class, "template_id": template_index,
                    "text": text, "tsne_x": float(coordinate[0]), "tsne_y": float(coordinate[1]),
                    "image_path": samples[image_index]["path"],
                })
        for group in ("empty", "subject"):
            for template_index, text in enumerate(texts[group]):
                coordinate = xyz[group][image_index, template_index]
                point_rows.append({
                    "condition": group, "image_class": image_class,
                    "text_class": "shared", "template_id": template_index,
                    "text": text, "tsne_x": float(coordinate[0]), "tsne_y": float(coordinate[1]),
                    "image_path": samples[image_index]["path"],
                })
    write_csv(out / "points.csv",
              ["condition", "image_class", "text_class", "template_id", "text",
               "tsne_x", "tsne_y", "image_path"], point_rows)
    np.savez_compressed(
        out / "features_and_centers.npz",
        image_classes=np.asarray(image_classes),
        candidate_classes=np.asarray(candidate_classes),
        z_class_raw=raw_z["class"], z_class_unit=unit(raw_z["class"]),
        z_empty_raw=raw_z["empty"], z_empty_unit=unit(raw_z["empty"]),
        z_subject_raw=raw_z["subject"], z_subject_unit=unit(raw_z["subject"]),
        tsne_class=xyz["class"], tsne_empty=xyz["empty"], tsne_subject=xyz["subject"],
        class_centers=unit(anchors), class_centers_tsne=center_xy,
        visual=visual, causal_e=causal,
    )
    metadata.update({
        "image_classes": image_classes, "candidate_classes": candidate_classes,
        "image_count": len(image_classes), "template_count": 80,
        "fusion_points": len(point_rows), "classification_anchor_count": len(candidate_classes),
        "z_class_shape": list(raw_z["class"].shape),
        "z_empty_shape": list(raw_z["empty"].shape),
        "z_subject_shape": list(raw_z["subject"].shape),
        "z_definition": "actual TDA(E, normalized CLIP environment text), after TDA LayerNorm",
        "class_condition": "each image uses every candidate class bank, with all 80 templates",
        "labels_in_fusion": False,
        "correct_class_center": "trained normalized classification text feature; recovered PromptLearner and CLIPTextEncoder",
        "center_coordinates": "classification anchors included as separate points in the same t-SNE fit",
        "overview_colors": "input image classes",
        "per_image_class_condition_colors": "candidate environment text classes",
    })
    (out / "run_info.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("\nClassification text anchor coordinates:")
    for row in centers_rows:
        print("{:<12} ({:9.4f}, {:9.4f})".format(
            row["class"], row["center_x"], row["center_y"]))


def restore_classification_anchors(prompt, model, tokenize, state, hparams, class_names, device):
    """Exactly the branch's trained class_features() computation."""
    import torch
    import torch.nn.functional as F

    # The branch's PromptLearner constructor creates CPU token tensors and
    # CPU context padding. Keep CLIP on CPU until initialization is complete.
    model.cpu()
    learned_state = substate(state, "text_features.prompt_learner.")
    if "context" not in learned_state:
        raise ValueError("Checkpoint is missing trained text_features.prompt_learner.context")
    context_length = int(learned_state["context"].shape[-2])
    learner = prompt.PromptLearner(
        class_names, model, tokenize, context_length,
        hparams.get("cipt_prompt_init", "a photo of a"),
    )
    learner.load_state_dict(learned_state, strict=True)
    model.to(device)
    learner.to(device).eval().requires_grad_(False)
    if learner.tokenized_prompts.shape[0] != len(class_names):
        raise ValueError("Checkpoint prompt count and checkpoint class names do not match")
    encoder = prompt.CLIPTextEncoder(model).to(device).eval()
    with torch.inference_mode():
        return F.normalize(encoder(learner(), learner.tokenized_prompts).float(), dim=-1)


def fuse_all_candidates(tda, e, candidate_banks):
    """No image class label is accepted or used by this inference operation."""
    num_candidates, templates, dim = candidate_banks.shape
    causal_flat = e.expand(num_candidates, dim)
    return tda(causal_flat, candidate_banks).reshape(num_candidates, templates, dim)


def main():
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn.functional as F

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    root = resolve_repo_root(args.repo_root)
    samples = select_images(args)
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(Path(args.checkpoint).expanduser(), **load_kwargs)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a DomainBed checkpoint dictionary")
    hparams = checkpoint.get("model_hparams", checkpoint.get("hparams", {}))
    saved_args = checkpoint.get("args", {})
    trained_dataset = saved_args.get("dataset") if isinstance(saved_args, dict) else getattr(saved_args, "dataset", None)
    if trained_dataset and trained_dataset != "PACS":
        raise ValueError("This script selects PACS images; checkpoint dataset is " + str(trained_dataset))
    if str(hparams.get("cipt_use_tda", True)).lower() in ("false", "0"):
        raise ValueError("Choose a checkpoint trained with TDA enabled")
    state = checkpoint_state(checkpoint)

    # Candidate names follow the checkpoint order; --classes selects images only.
    candidate_classes = list(hparams.get("cipt_class_names") or PACS_CLASSES)
    candidate_lookup = {canonical(name): i for i, name in enumerate(candidate_classes)}
    if len(candidate_lookup) != len(candidate_classes):
        raise ValueError("Checkpoint class names must be distinct")
    if any(canonical(name).startswith("class ") for name in candidate_classes):
        raise ValueError("Real checkpoint class names are required")
    for name in args.classes:
        if canonical(name) not in candidate_lookup:
            raise ValueError("Image class {} is not in checkpoint classes {}".format(name, candidate_classes))

    prompt = load_module("_fusion_tsne_prompt", root / "domainbed/algorithms/cipt_prompt.py")
    modules = load_module("_fusion_tsne_modules", root / "domainbed/algorithms/cipt_modules.py")
    transforms = load_module("_fusion_tsne_transforms", root / "domainbed/datasets/transforms.py")
    model, tokenize = prompt.load_frozen_clip(
        hparams.get("cipt_clip_backbone", "ViT-B/16"), args.clip_path
    )
    clip_state = substate(state, "clip_model.")
    if clip_state:
        model.load_state_dict(clip_state, strict=True)
    # restore_classification_anchors initializes CPU prompt buffers first,
    # then moves CLIP and the restored prompts to the requested device.
    model.eval().requires_grad_(False)
    dim = int(model.visual.output_dim)
    decomposition = modules.CausalDecomposition(dim)
    decomposition.load_state_dict(substate(state, "causal_decomposition."), strict=True)
    decomposition.to(device).eval().requires_grad_(False)
    tda = modules.TextDiversityAugmentation(dim, int(hparams.get("cipt_tda_heads", 1)))
    tda.load_state_dict(substate(state, "tda."), strict=True)
    tda.to(device).eval().requires_grad_(False)

    anchors = restore_classification_anchors(
        prompt, model, tokenize, state, hparams, candidate_classes, device
    )
    templates = list(prompt.B5B_IMAGENET_TEMPLATES)
    if len(templates) != 80:
        raise ValueError("Expected the branch's 80 B5B templates")
    texts = {
        "class": [[template.format(name.replace("_", " ")) for template in templates]
                  for name in candidate_classes],
        "empty": [template.format("") for template in templates],
        "subject": [template.format("subject") for template in templates],
    }

    def encode(text_list):
        chunks = []
        for start in range(0, len(text_list), 64):
            tokens = tokenize(text_list[start:start + 64]).to(device)
            chunks.append(F.normalize(model.encode_text(tokens).float(), dim=-1))
        return torch.cat(chunks)

    fused = {name: [] for name in GROUPS}
    visual_features, causal_features = [], []
    with torch.inference_mode():
        banks = {
            "class": encode([text for bank in texts["class"] for text in bank]).reshape(
                len(candidate_classes), 80, dim),
            "empty": encode(texts["empty"]),
            "subject": encode(texts["subject"]),
        }
        for sample in samples:
            with Image.open(sample["path"]) as image:
                image_tensor = transforms.clip_basic(image).unsqueeze(0).to(device)
            visual = F.normalize(model.encode_image(image_tensor).float(), dim=-1)
            e, _ = decomposition(visual)
            visual_features.append(visual[0].cpu().numpy())
            causal_features.append(e[0].cpu().numpy())

            # All candidates are fused before the ground-truth annotation is read.
            z_all = fuse_all_candidates(tda, e, banks["class"])
            fused["class"].append(z_all.cpu().numpy())
            for group in ("empty", "subject"):
                z = tda(e, banks[group])
                if tuple(z.shape) != (1, 80, dim):
                    raise ValueError("Unexpected shared TDA output shape: {}".format(tuple(z.shape)))
                fused[group].append(z[0].cpu().numpy())
            print("{} image: {} candidate classes x 80 + empty 80 + subject 80 | {}".format(
                sample["class"], len(candidate_classes), sample["path"]), flush=True)

    raw_z = {group: np.stack(fused[group]) for group in GROUPS}
    anchor_array = anchors.cpu().numpy()
    flat_features, counts = flatten_features(raw_z, anchor_array)
    flat_xy = project(flat_features, args.perplexity, args.seed)
    xyz, center_xy = split_coordinates(flat_xy, raw_z, counts, len(candidate_classes))
    out = Path(args.output_dir).expanduser().resolve()
    save_results(
        out, raw_z, xyz, anchor_array, center_xy, args.classes, candidate_classes,
        samples, texts, np.stack(visual_features), np.stack(causal_features),
        {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "clip_path": str(Path(args.clip_path).expanduser().resolve()),
            "repo_root": str(root), "domain": args.domain, "seed": args.seed,
            "perplexity": args.perplexity, "device": device,
            "trained_template_mode": hparams.get("cipt_template_mode"),
            "checkpoint_hparams": hparams,
            "preprocessing": "branch clip_basic; normalized CLIP visual -> causal adapter -> TDA",
            "protocol": "candidate-class inference construction, expanded to all 80 templates for visualization",
        },
    )
    print("\nSaved plots and correct class anchors to {}".format(out))


if __name__ == "__main__":
    main()
