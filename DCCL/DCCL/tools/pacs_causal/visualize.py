"""One PCA fit and one t-SNE fit per held-out fold, with paired sample IDs."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


KEYS = ("z", "E_beta", "S_removed", "E_full_res", "E_orth")
DISPLAY = ("Original z", "DICM E (beta = 0.75)", "Removed S", "Full residual removal", "Orthogonal removal")
COLORS = ("#17659b", "#bf5223", "#198168", "#9254ab", "#ad2463", "#88701f", "#4b566d")
MARKERS = ("o", "^", "s", "D")


def plots(payload, directory):
    directory.mkdir(parents=True, exist_ok=True)
    idx = payload["tsne"]
    data = np.concatenate([payload["representations"][key][idx] for key in KEYS])
    # Center and scale the full joint cloud together. Keep differences in feature energy.
    centered = data - data.mean(0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.einsum("nd,nd->n", centered, centered))))
    inputs = centered / max(scale, 1e-12)
    pca = PCA(n_components=min(50, len(inputs)-1, inputs.shape[1]), svd_solver="randomized",
              random_state=19)
    low = pca.fit_transform(inputs)
    settings = dict(n_components=2, perplexity=30, max_iter=1000, learning_rate="auto",
                    random_state=19, init="pca", n_jobs=2)
    estimator = TSNE(**settings)
    flat = estimator.fit_transform(low)
    xy = flat.reshape(len(KEYS), len(idx), 2)
    domain, label = payload["env"][idx], payload["y"][idx]
    domains, classes = payload["domains"], payload["classes"]
    lo, hi = xy.min(axis=(0, 1)), xy.max(axis=(0, 1))
    padding = (hi - lo) * .055
    limits = [[float(lo[i]-padding[i]), float(hi[i]+padding[i])] for i in range(2)]

    with (directory / "tsne_coordinates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "source_domain", "category", "view", "tsne_x", "tsne_y"])
        for j, view in enumerate(KEYS):
            for k, source_i in enumerate(idx):
                writer.writerow([payload["source"][source_i]["sample_id"], domains[domain[k]],
                                 classes[label[k]], view, *xy[j, k].tolist()])
    config = dict(seed=19, pca_components=pca.n_components_, pca_explained_variance=float(
        pca.explained_variance_ratio_.sum()), tsne=settings, kl_divergence=float(estimator.kl_divergence_),
        views=KEYS, class_colors=dict(zip(classes, COLORS)), domain_markers=dict(zip(domains, MARKERS)),
        shared_limits=limits, common_scale=scale, sample_count=len(idx))
    (directory / "plot_settings.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def draw(ax, j, only=None):
        for y in np.unique(label):
            for d in np.unique(domain):
                if only is not None and d != only:
                    continue
                keep = (label == y) & (domain == d)
                ax.scatter(*xy[j, keep].T, s=8.5, c=COLORS[y], marker=MARKERS[d],
                           alpha=.69, edgecolors="none", rasterized=True)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d1d9e0")

    class_handles = [Line2D([], [], color=color, marker="o", ls="", label=label)
                     for label, color in zip(classes, COLORS)]
    domain_handles = [Line2D([], [], color="#424b52", marker=MARKERS[d], ls="", label=domains[d])
                      for d in np.unique(domain)]
    target = domains[payload["target"]].replace("_", " ").title()

    def save(fig, name, subtitle, handles):
        fig.suptitle(f"PACS  |  Held-out {target}", color="#202833",
                     fontsize=15, fontweight="semibold", y=.99)
        fig.text(.5, .92 if name != "domain_views" else .95, subtitle,
                 ha="center", fontsize=9, color="#465565")
        fig.legend(handles=handles, loc="lower center", frameon=False,
                   ncol=min(len(handles), 7), fontsize=8, bbox_to_anchor=(.5, .01))
        for ext in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{ext}", dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    for name, ids, label_text in (
            ("category_views", (0, 1, 2), "Source validation samples, shared t-SNE embedding"),
            ("removal_views", (1, 3, 4), "Residual attenuation, full residual removal, and orthogonal removal")):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
        fig.subplots_adjust(left=.035, right=.98, bottom=.16, top=.82, wspace=.10)
        for ax, j in zip(axes, ids):
            draw(ax, j)
            ax.set_title(DISPLAY[j], color="#24313e", fontsize=10)
        save(fig, name, label_text, class_handles + domain_handles)

    observed_domains = np.unique(domain)
    fig, axes = plt.subplots(len(observed_domains), 3, figsize=(10.3, 10.5))
    fig.subplots_adjust(left=.14, right=.98, bottom=.08, top=.90, wspace=.06, hspace=.09)
    for r, d in enumerate(observed_domains):
        for c in range(3):
            draw(axes[r, c], c, only=d)
            if r == 0:
                axes[r, c].set_title(DISPLAY[c], fontsize=10)
        axes[r, 0].set_ylabel(domains[d].replace("_", " ").title(), fontsize=9)
    save(fig, "domain_views", "Each row isolates one source domain on the same axes", class_handles)
    return config
