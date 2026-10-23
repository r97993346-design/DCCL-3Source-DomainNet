#!/usr/bin/env python3
"""Plot same-class E/S features from a trained CIPTDCCL checkpoint, BEFORE TDA.

Run from the directory containing train_all.py. Uses the repository's frozen
CLIP preprocessing and the checkpoint's two linear decomposition adapters.
No training, class prompts, TDA, artificial views, or dataset splits are run.
Each class gets one jointly fitted E/S embedding with three shared-axis panels.
--samples-per-class-domain 0 uses every image of the selected class/domain.
"""

import argparse
import csv
import inspect
import json
import os
from pathlib import Path
import random
import sys

import numpy as np


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PACS_ALIASES = {
    "a": "artpainting", "artpainting": "artpainting", "painting": "artpainting",
    "c": "cartoon", "cartoon": "cartoon", "p": "photo", "photo": "photo",
    "s": "sketch", "sketch": "sketch",
}


def token(name):
    return "".join(c for c in str(name).lower() if c.isalnum())


def selected_names(requested, available, dataset, domain=False):
    if [x.lower() for x in requested] == ["all"]:
        return list(available)
    def key(value):
        value = token(value)
        return PACS_ALIASES.get(value, value) if domain and dataset == "PACS" else value
    lookup = {key(x): x for x in available}
    result = []
    for item in requested:
        if key(item) not in lookup:
            raise ValueError("Unknown {} {!r}; available: {}".format(
                "domain" if domain else "class", item, ", ".join(available)))
        actual = lookup[key(item)]
        if actual not in result:
            result.append(actual)
    return result


def find_dataset_root(data_dir, dataset):
    root = Path(data_dir).expanduser().resolve()
    folder = {"PACS": "PACS", "VLCS": "VLCS", "OfficeHome": "office_home",
              "DomainNet": "domain_net"}[dataset]
    candidates = [root / folder, root / dataset, root]
    for parent in list(candidates):
        candidates.extend([parent / "kfold", parent / "images"])
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_dir():
            continue
        for domain in sorted(candidate.iterdir()):
            if not domain.is_dir():
                continue
            for cls in domain.iterdir():
                if cls.is_dir() and any(p.is_file() and p.suffix.lower() in EXTENSIONS
                                        for p in cls.iterdir()):
                    return candidate
    raise FileNotFoundError("Cannot find domain/class/image folders below {}".format(root))


def collect_records(args):
    root = find_dataset_root(args.data_dir, args.dataset)
    available_domains = sorted(p.name for p in root.iterdir() if p.is_dir())
    domains = selected_names(args.domains, available_domains, args.dataset, domain=True)
    available_classes = sorted({p.name for d in domains for p in (root / d).iterdir()
                                if p.is_dir()})
    classes = selected_names(args.classes, available_classes, args.dataset)
    rng = np.random.default_rng(args.seed)
    records = []
    for cls in classes:
        for domain in domains:
            folder = root / domain / cls
            files = sorted(p for p in folder.rglob("*")
                           if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if not files:
                raise FileNotFoundError("No images: {}".format(folder))
            if args.samples_per_class_domain > 0 and len(files) > args.samples_per_class_domain:
                indices = sorted(rng.choice(len(files), args.samples_per_class_domain,
                                            replace=False).tolist())
                files = [files[i] for i in indices]
            records.extend((str(p), domain, cls) for p in files)
            print("{} / {}: {} images".format(cls, domain, len(files)), flush=True)
    return records, root, available_domains


class ImageRecords:
    def __init__(self, records, preprocess):
        self.records, self.preprocess = records, preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        path = self.records[index][0]
        try:
            with Image.open(path) as image:
                return self.preprocess(image.convert("RGB")), index
        except Exception as exc:
            raise RuntimeError("Cannot read {}: {}".format(path, exc)) from exc


def adapter_parameters(state):
    result = {}
    prefix = None
    for branch in ("causal", "spurious"):
        for part in ("weight", "bias"):
            suffix = "causal_decomposition.{}_adapter.{}".format(branch, part)
            matches = [k for k in state if k == suffix or k.endswith("." + suffix)]
            if len(matches) != 1:
                raise ValueError("Expected exactly one checkpoint tensor ending in {!r}, got {}".
                                 format(suffix, matches))
            name = matches[0]
            current_prefix = name[:-len(suffix)]
            if prefix is not None and prefix != current_prefix:
                raise ValueError("E/S tensors have inconsistent model prefixes")
            prefix = current_prefix
            result["{}_adapter.{}".format(branch, part)] = state[name]
    return result, prefix


def extract_features(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    load_options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = False
    checkpoint = torch.load(checkpoint_path, **load_options)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a dictionary checkpoint")
    state_key = args.state_key
    if state_key is None:
        state_key = next((k for k in ("model_dict", "state_dict", "model_state_dict")
                          if isinstance(checkpoint.get(k), dict)), None)
    state = checkpoint[state_key] if state_key else checkpoint
    params, prefix = adapter_parameters(state)
    hparams = checkpoint.get("model_hparams", checkpoint.get("hparams", {}))
    if not isinstance(hparams, dict):
        hparams = vars(hparams)
    if hparams.get("cipt_use_de") is False:
        raise ValueError("This checkpoint was trained with cipt_use_de=false")
    clip_path = Path(args.clip_path or hparams.get("cipt_clip_path", "")).expanduser()
    if not clip_path.is_file():
        raise FileNotFoundError("Supply --clip-path pointing to the local CLIP .pt file")

    # Find the vendored OpenAI CLIP package without importing the full algorithm.
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        bundled = parent / "CLIP"
        if (bundled / "clip" / "__init__.py").is_file():
            sys.path.insert(0, str(bundled))
            break
    import clip
    if not hasattr(clip, "load"):
        raise ImportError("Expected the repository's OpenAI CLIP package")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(str(clip_path), device="cpu", jit=False)
    model.float()
    visual_prefix = prefix + "clip_model.visual."
    visual_state = {k[len(visual_prefix):]: v for k, v in state.items()
                    if k.startswith(visual_prefix)}
    if visual_state:
        model.visual.load_state_dict(visual_state, strict=True)
    model.requires_grad_(False).to(device).eval()
    dim = int(model.visual.output_dim)
    adapters = {}
    for branch in ("causal", "spurious"):
        layer = torch.nn.Linear(dim, dim)
        layer.load_state_dict({p: params["{}_adapter.{}".format(branch, p)]
                               for p in ("weight", "bias")}, strict=True)
        adapters[branch] = layer.requires_grad_(False).to(device).eval()
    records, root, domain_order = collect_records(args)
    loader = DataLoader(ImageRecords(records, preprocess), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers,
                        pin_memory=str(device).startswith("cuda"))
    values = {"V": [], "E": [], "S": []}
    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(loader):
            # Exactly matches CIPTDCCL._visual(), followed by CausalDecomposition.
            visual = F.normalize(model.encode_image(images.to(device)).float(), dim=-1)
            values["V"].append(visual.cpu().numpy())
            values["E"].append(adapters["causal"](visual).cpu().numpy())
            values["S"].append(adapters["spurious"](visual).cpu().numpy())
            if batch_index % 10 == 0 or batch_index + 1 == len(loader):
                print("Extracted batch {}/{}".format(batch_index + 1, len(loader)), flush=True)
    features = {k: np.concatenate(v, axis=0) for k, v in values.items()}
    features.update(paths=np.asarray([r[0] for r in records]),
                    domains=np.asarray([r[1] for r in records]),
                    classes=np.asarray([r[2] for r in records]))
    metadata = {"checkpoint": str(checkpoint_path), "state_key": state_key or "root",
                "adapter_prefix": prefix, "clip_path": str(clip_path.resolve()),
                "clip_visual_restored_from_checkpoint": bool(visual_state),
                "dataset_root": str(root), "domain_order": domain_order,
                "test_envs": checkpoint.get("test_envs", []),
                "feature_stage": "raw linear E/S outputs before TDA; CLIP input L2-normalized"}
    return features, metadata


def plot_class(features, cls, args, output_dir):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    indices = np.flatnonzero(features["classes"] == cls)
    e, s = features["E"][indices], features["S"][indices]
    if e.ndim != 2 or s.shape != e.shape or len(e) < 3:
        raise ValueError("{} needs at least 3 paired E/S image vectors".format(cls))
    x = np.concatenate([e, s]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite E/S features in {}".format(cls))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if args.normalization == "l2":
        x = x / np.maximum(norms, 1e-12)
    pca_variance = None
    components = min(args.pca_components, x.shape[1], len(x) - 1)
    if 0 < components < x.shape[1]:
        pca = PCA(n_components=components, svd_solver="full")
        x = pca.fit_transform(x)
        pca_variance = float(pca.explained_variance_ratio_.sum())
    perplexity = min(args.perplexity, max(1.0, (len(x) - 1) / 3.0))
    kwargs = dict(n_components=2, perplexity=perplexity, init="pca",
                  learning_rate=200.0, random_state=args.seed, metric="euclidean")
    iteration_key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    kwargs[iteration_key] = args.iterations
    print("t-SNE {}: {} E + {} S points, perplexity={:.2f}".
          format(cls, len(e), len(s), perplexity), flush=True)
    xy = TSNE(**kwargs).fit_transform(x)  # ONE fit, shared by ALL three panels.
    e_xy, s_xy = np.split(xy, 2)
    domains = features["domains"][indices]
    domain_names = sorted(set(domains.tolist()))
    palette = plt.get_cmap("tab10")
    colors = {name: palette(i % 10) for i, name in enumerate(domain_names)}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, title, branches in zip(axes, ["E", "S", "E + S"],
                                   [(0,), (1,), (0, 1)]):
        for branch in branches:
            points = (e_xy, s_xy)[branch]
            for domain in domain_names:
                mask = domains == domain
                ax.scatter(points[mask, 0], points[mask, 1], s=13, alpha=.65,
                           marker=("o", "^")[branch], color=colors[domain],
                           linewidths=0, rasterized=True)
        ax.set_title("{}: {} images".format(title, len(e)))
        ax.set_xlabel("t-SNE 1")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.12)
    axes[0].set_ylabel("t-SNE 2")
    handles = [Line2D([], [], marker="o", linestyle="", color=colors[d],
                      label="{} (n={})".format(d, int((domains == d).sum())))
               for d in domain_names]
    handles.extend([Line2D([], [], marker="o", linestyle="", color="black", label="E"),
                    Line2D([], [], marker="^", linestyle="", color="black", label="S")])
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False)
    fig.suptitle("{} | decomposition BEFORE TDA | {} features\n"
                 "One shared embedding for E, S and the overlay".format(cls, args.normalization))
    fig.tight_layout(rect=(0, .09, 1, .90))
    stem = "{}_es_tsne".format(cls)
    fig.savefig(output_dir / (stem + ".png"), dpi=220)
    fig.savefig(output_dir / (stem + ".pdf"))
    plt.close(fig)
    np.savez_compressed(output_dir / (stem + "_coordinates.npz"), E_xy=e_xy, S_xy=s_xy,
                        paths=features["paths"][indices], domains=domains)
    with (output_dir / (stem + "_coordinates.csv")).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "domain", "E_x", "E_y", "S_x", "S_y", "E_norm", "S_norm"])
        for j, original_index in enumerate(indices):
            writer.writerow([features["paths"][original_index], domains[j], *e_xy[j], *s_xy[j],
                             float(norms[j, 0]), float(norms[len(e) + j, 0])])
    return {"class": cls, "images": len(e), "perplexity": perplexity,
            "pca_retained_variance": pca_variance,
            "domain_counts": {d: int((domains == d).sum()) for d in domain_names},
            "E_norm_mean": float(norms[:len(e)].mean()),
            "S_norm_mean": float(norms[len(e):].mean()),
            "near_zero_vectors": int((norms < 1e-12).sum()),
            "embedding": "ONE joint fit on [E; S]; all panels share coordinates"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="trained CIPTDCCL .pth")
    source.add_argument("--features", help="replot an existing features_es.npz; does not need torch/CLIP")
    parser.add_argument("--state-key", default=None, help="explicit checkpoint dictionary key")
    parser.add_argument("--dataset", choices=["PACS", "VLCS", "OfficeHome", "DomainNet"], default="PACS")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--clip-path", default=None)
    parser.add_argument("--classes", nargs="+", default=["dog"], help="class folder names, or all")
    parser.add_argument("--domains", nargs="+", default=["all"], help="domain folder names, or all")
    parser.add_argument("--samples-per-class-domain", type=int, default=0, help="0 = all images")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pca-components", type=int, default=50, help="0 disables joint PCA")
    parser.add_argument("--normalization", choices=["l2", "raw"], default="l2")
    parser.add_argument("--output-dir", default="train_output/ablation/es_tsne")
    args = parser.parse_args()
    if args.samples_per_class_domain < 0 or args.batch_size < 1 or args.workers < 0:
        parser.error("sample cap/workers must be nonnegative and batch size positive")
    if args.iterations < 250 or args.perplexity <= 0 or args.pca_components < 0:
        parser.error("iterations >= 250, perplexity > 0, pca-components >= 0 required")
    if args.checkpoint and not args.data_dir:
        parser.error("--data-dir is required with --checkpoint")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.features:
        with np.load(Path(args.features).expanduser(), allow_pickle=False) as archive:
            features = {k: archive[k] for k in ("E", "S", "paths", "domains", "classes")}
        if len({len(v) for v in features.values()}) != 1:
            raise ValueError("Feature cache has inconsistent array lengths")
        requested_domains = selected_names(args.domains, sorted(set(features["domains"])),
                                           args.dataset, domain=True)
        mask = np.isin(features["domains"], requested_domains)
        features = {k: v[mask] for k, v in features.items()}
        metadata = {"feature_cache": str(Path(args.features).expanduser().resolve())}
    else:
        features, metadata = extract_features(args)
    classes = selected_names(args.classes, sorted(set(features["classes"])), args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.features:
        np.savez_compressed(output_dir / "features_es.npz", **features)
    summaries = [plot_class(features, cls, args, output_dir) for cls in classes]
    metadata.update(settings=vars(args), classes=summaries,
                    note="Descriptive visualization; 2D distances do not prove causal purity or uniform logits")
    (output_dir / "metadata_es.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved {} class plot(s) to {}".format(len(summaries), output_dir), flush=True)


if __name__ == "__main__":
    main()
