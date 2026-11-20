"""PACS performance, directional gates, and shared source-validation embeddings."""

from itertools import product
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tools.pacs_causal.visualize import COLORS, MARKERS
from tools.pacs_directdiff.common import DOMAINS, GROUPS, ROOT


LABELS = ("Art Painting", "Cartoon", "Photo", "Sketch")
SETUPS = ("P0", "A16", "Uniform", "Gate")
PALETTE = ("#667788", "#6a57a5", "#187c76", "#b66a22")


def save_figure(fig, path):
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def paired_tsne(domain, variant):
    """One PCA and one t-SNE fit across both checkpoints' matched z/E/S samples."""
    views = "z", "E", "S"
    output = ROOT / "visualizations" / domain / variant
    output.mkdir(parents=True, exist_ok=True)
    arrays, sample = [], None
    for group in GROUPS:
        path = ROOT / "features" / group / domain / variant / "features.npz"
        with np.load(path, allow_pickle=False) as bank:
            selected = bank["tsne_indices"]
            info = dict(sample_id=bank["sample_id"][selected], y=bank["y"][selected],
                        domain=bank["domain"][selected], classes=bank["classes"], domains=bank["domains"])
            if sample is None:
                sample = info
            else:
                for name in info:
                    np.testing.assert_array_equal(info[name], sample[name])
            assert domain not in info["domains"][info["domain"]]
            arrays.extend(bank[key][selected] for key in views)
    joint = np.concatenate(arrays)
    centered = joint - joint.mean(0)
    rms = float(np.sqrt(np.mean(np.einsum("nd,nd->n", centered, centered))))
    if rms <= 1e-12:
        raise ValueError("Collapsed source-validation rendering features")
    pca = PCA(n_components=50, svd_solver="randomized", random_state=19)
    reduced = pca.fit_transform(centered / rms)
    estimator = TSNE(n_components=2, perplexity=30, max_iter=1000, init="pca",
                     learning_rate="auto", n_jobs=2, random_state=19)
    xy = estimator.fit_transform(reduced).reshape(len(GROUPS), len(views), len(sample["y"]), 2)
    lo, hi = xy.min(axis=(0, 1, 2)), xy.max(axis=(0, 1, 2))
    pad = (hi - lo) * .05
    limits = [(lo[i] - pad[i], hi[i] + pad[i]) for i in range(2)]
    with (output / "tsne_coordinates.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("group", "checkpoint", "view", "sample_id", "category", "source_domain", "x", "y"))
        for (i, group), (j, view) in product(enumerate(GROUPS), enumerate(views)):
            for sid, c, d, point in zip(sample["sample_id"], sample["y"], sample["domain"], xy[i, j]):
                writer.writerow((group, variant, view, sid, sample["classes"][c],
                                 sample["domains"][d], *point))
    metadata = dict(held_out=domain, checkpoint=variant, source_validation_samples=len(sample["y"]),
                    shared_representations=6, scaling_rms=rms, seed=19, perplexity=30, iterations=1000,
                    pca_components=50, pca_variance=float(pca.explained_variance_ratio_.sum()),
                    kl=float(estimator.kl_divergence_), axis_limits=[list(map(float, x)) for x in limits])
    (output / "plot_settings.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    for label, key in (("category", "y"), ("source_domain", "domain")):
        figure, axes = plt.subplots(2, 3, figsize=(11.8, 8.7))
        figure.subplots_adjust(left=.105, top=.86, bottom=.10, right=.98, wspace=.07, hspace=.14)
        color_id = sample[key]
        names = sample["classes"] if key == "y" else sample["domains"]
        for (i, group), (j, view) in product(enumerate(GROUPS), enumerate(views)):
            ax = axes[i, j]
            for k in np.unique(color_id):
                for s in np.unique(sample["domain"]):
                    pick = (color_id == k) & (sample["domain"] == s)
                    ax.scatter(*xy[i, j, pick].T, s=10, c=COLORS[k], marker=MARKERS[s],
                               alpha=.64, linewidths=0, rasterized=True)
            ax.set(xlim=limits[0], ylim=limits[1], xticks=[], yticks=[], aspect="equal")
            if i == 0:
                ax.set_title(("Original z", "Stable E", "Removed S")[j], fontsize=10, weight="bold")
            if j == 0:
                ax.set_ylabel(group, fontsize=11, weight="bold")
            for spine in ax.spines.values():
                spine.set_color("#d5dde1")
        handles = [plt.Line2D([], [], color=COLORS[k], marker="o", ls="", label=names[k])
                   for k in np.unique(color_id)]
        figure.legend(handles=handles, ncol=min(7, len(handles)), loc="lower center", frameon=False)
        figure.suptitle(f"PACS  |  {LABELS[DOMAINS.index(domain)]} held out  |  {variant}",
                         fontsize=15, weight="bold", y=.98, color="#233641")
        figure.text(.5, .921, "Matched source validation images on one joint embedding",
                    ha="center", color="#516675", fontsize=9)
        save_figure(figure, output / label)


def performance_figure(scores):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    fig.subplots_adjust(left=.08, right=.98, top=.80, bottom=.22, wspace=.22)
    fig.suptitle("PACS  |  DirectDiff16 target-domain accuracy", x=.08, y=.96, ha="left",
                 weight="bold", color="#233641", fontsize=16)
    for ax, key in zip(axes, ("swad_test_in", "swad_test_out")):
        for group, color in zip(SETUPS, PALETTE):
            folds = [next((r for r in scores if (r["group"], r["domain"]) == (group, domain)), None)
                     for domain in DOMAINS]
            ys = [100 * row[key] if row is not None else np.nan for row in folds]
            ax.plot(range(4), ys, "o-", label=group, c=color, markersize=4, linewidth=1.6)
            for i, row in enumerate(folds):
                if row is not None and not row["swad_converged"]:
                    ax.scatter(i, ys[i], marker="x", color=color, s=65, zorder=5)
        ax.set_title(key.replace("_", " "), loc="left", weight="bold")
        ax.set_xticks(range(4), LABELS, rotation=18)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", color="#dbe5e9")
        ax.spines[["right", "top"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    save_figure(fig, ROOT / "reports" / "swad_performance")


def gate_figure(rows):
    weights = np.asarray([r["gate"]["gains"] for r in rows])
    fig, ax = plt.subplots(figsize=(10.6, 3.7))
    fig.subplots_adjust(left=.21, top=.82, bottom=.16, right=.90)
    figure = ax.imshow(weights, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(figure, ax=ax, label="Attenuation g", fraction=.07)
    ax.set_xticks(range(16), range(1, 17))
    ax.set_yticks(range(len(rows)), [f'{r["held_out"]} · {r["checkpoint"]}' for r in rows], fontsize=8)
    ax.set_xlabel("Direction ordered by the domain-minus-class eigenvalue")
    ax.set_title("PACS  |  Estimated DirectDiff16 gate coefficients", loc="left", weight="bold")
    save_figure(fig, ROOT / "reports" / "gate_coefficients")
