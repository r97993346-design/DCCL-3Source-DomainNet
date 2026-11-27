#!/usr/bin/env python3
"""Plot CLIP text features: class names, empty class slots, and subject.

Run in the existing DomainBed/CLIP environment:
    python tsne_clip_text.py --clip-path /path/to/ViT-B-16.pt

Default templates are the 80 templates in OpenAI CLIP's official notebook:
https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
To use your exact repository templates, pass --templates-file templates.txt
(one template per line, with {} or {class} as the class slot).
This script extracts frozen CLIP text features only; no training is performed.
"""

import argparse
import csv
import inspect
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-path", required=True, help="Local OpenAI CLIP .pt weight file")
    parser.add_argument("--classes", nargs="+", default=PACS_CLASSES)
    parser.add_argument("--templates-file", help="Optional UTF-8 file: one template per line")
    parser.add_argument("--output-dir", default="train_output/ablation/text_tsne_80")
    parser.add_argument("--device", default=None, help="Default: cuda when available, otherwise cpu")
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_templates(filename):
    if filename is None:
        return TEMPLATES
    templates = [
        line.strip()
        for line in Path(filename).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not templates:
        raise ValueError("The template file is empty.")
    for template in templates:
        if "{}" not in template and "{class}" not in template:
            raise ValueError(f"Template has no {{}} or {{class}} slot: {template}")
    return templates


def fill_template(template, subject):
    return template.replace("{}", subject).replace("{class}", subject)


def build_texts(templates, classes):
    rows = []
    for class_name in classes:
        for index, template in enumerate(templates):
            rows.append(("class", class_name, index, fill_template(template, class_name)))
    for group, subject in [("empty", ""), ("subject", "subject")]:
        for index, template in enumerate(templates):
            # Encode each shared template once, without assigning class labels.
            rows.append((group, "", index, fill_template(template, subject)))
    return rows


def encode_texts(texts, clip_path, device):
    import torch
    import clip

    if not hasattr(clip, "load") or not hasattr(clip, "tokenize"):
        raise ImportError("This script requires OpenAI CLIP, not the unrelated PyPI 'clip' package.")
    if not Path(clip_path).is_file():
        raise FileNotFoundError(f"CLIP weight file does not exist: {clip_path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = clip.load(clip_path, device=device, jit=False)
    model.eval()
    encoded = []
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            tokens = clip.tokenize(texts[start:start + 64]).to(device)
            features = model.encode_text(tokens).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            encoded.append(features.cpu().numpy())
    return np.concatenate(encoded, axis=0)


def project(features, perplexity, seed):
    if not 0 < perplexity < len(features):
        raise ValueError(f"perplexity must be between 0 and {len(features)}.")
    kwargs = dict(
        n_components=2,
        metric="cosine",
        init="pca",
        learning_rate=200.0,
        perplexity=perplexity,
        random_state=seed,
    )
    # Support both older and newer scikit-learn versions.
    iteration_arg = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    kwargs[iteration_arg] = 1500
    return TSNE(**kwargs).fit_transform(features)


def draw_figures(xy, rows, classes, output_dir):
    group = np.array([row[0] for row in rows])
    labels = np.array([row[1] for row in rows])
    colors = plt.get_cmap("tab10")
    class_colors = {name: colors(i % 10) for i, name in enumerate(classes)}

    def scatter_classes(ax):
        for name in classes:
            selected = (group == "class") & (labels == name)
            ax.scatter(
                xy[selected, 0], xy[selected, 1],
                s=23, color=class_colors[name], alpha=0.8, label=name,
                edgecolors="none",
            )

    x_pad = max(float(np.ptp(xy[:, 0])) * 0.06, 1.0)
    y_pad = max(float(np.ptp(xy[:, 1])) * 0.06, 1.0)
    x_limits = (float(xy[:, 0].min()) - x_pad, float(xy[:, 0].max()) + x_pad)
    y_limits = (float(xy[:, 1].min()) - y_pad, float(xy[:, 1].max()) + y_pad)

    def style(ax, title):
        ax.set_title(title, fontsize=12)
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    scatter_classes(axes[0])
    axes[0].legend(fontsize=8, frameon=False)
    style(axes[0], f"Class-specific texts (n={np.sum(group == 'class')})")
    for ax, name, color, title in [
        (axes[1], "empty", "#52616b", "Empty class slot"),
        (axes[2], "subject", "#9864bd", "Class replaced by subject"),
    ]:
        selected = group == name
        ax.scatter(xy[selected, 0], xy[selected, 1], s=28, color=color, alpha=0.85)
        style(ax, f"{title} (n={np.sum(selected)})")
    fig.suptitle("CLIP text features — one joint t-SNE embedding", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_three_groups.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter_classes(ax)
    for name, color, marker, title in [
        ("empty", "#222222", "x", "Empty class slot"),
        ("subject", "#9864bd", "^", "Subject"),
    ]:
        selected = group == name
        ax.scatter(
            xy[selected, 0], xy[selected, 1],
            s=42, color=color, marker=marker, alpha=0.9, label=title,
        )
    style(ax, "CLIP text features: class / empty / subject")
    ax.legend(fontsize=9, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_joint.png", dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    templates = load_templates(args.templates_file)
    rows = build_texts(templates, args.classes)
    print(f"Templates: {len(templates)}; classes: {len(args.classes)}; texts: {len(rows)}")
    features = encode_texts([row[3] for row in rows], args.clip_path, args.device)
    # Fit once on all three groups and reuse the coordinates for every figure.
    xy = project(features, args.perplexity, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_figures(xy, rows, args.classes, output_dir)

    with (output_dir / "texts_and_coordinates.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["group", "class", "template_id", "text", "tsne_x", "tsne_y"])
        for row, coordinate in zip(rows, xy):
            writer.writerow([*row, *coordinate.tolist()])
    np.savez_compressed(output_dir / "text_features.npz", features=features, tsne=xy)
    print(f"Saved: {output_dir / 'tsne_three_groups.png'}")
    print(f"Saved: {output_dir / 'tsne_joint.png'}")


if __name__ == "__main__":
    main()

