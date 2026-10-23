"""Image-level CIPT diagnostics; all statistics use original unit embeddings.

The 80 template interventions are repeated measurements, never 80 independent
images. t-SNE coordinates are used only for illustration, not for metrics.
"""

import csv
import inspect
import json
from pathlib import Path

import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norms < 1e-12):
        raise ValueError("Expected finite, nonzero feature vectors.")
    return x / norms


def cosine_shift(a, b):
    return 1 - np.clip(np.sum(unit(a) * unit(b), axis=-1), -1, 1)


def template_spread(z):
    """Mean 1-cos over distinct template pairs, one result per image."""
    z = unit(z)
    k = z.shape[-2]
    if k < 2:
        raise ValueError("Need at least two templates.")
    return np.maximum(0, (k * k - np.square(z.sum(axis=-2)).sum(axis=-1)) / (k * (k - 1)))


def neighbor_weights(distance, k):
    """Fractional top-k membership handles tied neighbors without order bias."""
    distance = np.asarray(distance).copy()
    n = len(distance)
    k = min(k, n - 1)
    if k < 1:
        raise ValueError("Need at least two images and k >= 1.")
    np.fill_diagonal(distance, np.inf)
    cutoff = np.partition(distance, k - 1, axis=1)[:, k - 1:k]
    tied = np.isclose(distance, cutoff, atol=1e-10, rtol=0)
    closer = (distance < cutoff) & ~tied
    remaining = k - closer.sum(axis=1)
    return closer.astype(float) + tied * (remaining / tied.sum(axis=1))[:, None]


def cosine_distance_matrix(features):
    features = unit(features)
    distance = np.maximum(0, 1 - np.clip(features @ features.T, -1, 1))
    np.fill_diagonal(distance, 0)
    return distance


def domain_metrics(features, domains, k=5):
    """Call separately within each true class, never pool class/domain labels."""
    from sklearn.metrics import silhouette_score
    domains = np.asarray(domains)
    unique, counts = np.unique(domains, return_counts=True)
    if len(unique) < 2 or np.min(counts) < 2:
        raise ValueError("Each class needs >=2 domains and >=2 images per domain.")
    distance = cosine_distance_matrix(features)
    same = domains[:, None] == domains[None, :]
    upper = np.triu(np.ones_like(same), 1)
    weights = neighbor_weights(distance, k)
    actual_k = weights.sum(axis=1)
    chance = (same.sum(axis=1) - 1) / (len(domains) - 1)
    return {
        "images": len(domains), "domains": len(unique),
        "domain_silhouette": float(silhouette_score(distance, domains, metric="precomputed")),
        "same_domain_neighbor_fraction": float(((weights * same).sum(axis=1) / actual_k).mean()),
        "neighbor_chance": float(chance.mean()),
        "neighbor_excess": float((((weights * same).sum(axis=1) / actual_k) - chance).mean()),
        "domain_cosine_gap": float(distance[upper & ~same].mean() - distance[upper & same].mean()),
    }


def topology_metrics(a, b, k=5):
    from scipy.stats import spearmanr
    da, db = cosine_distance_matrix(a), cosine_distance_matrix(b)
    wa, wb = neighbor_weights(da, k), neighbor_weights(db, k)
    upper = np.triu_indices(len(a), 1)
    va, vb = da[upper], db[upper]
    rho = None
    if np.ptp(va) > 1e-12 and np.ptp(vb) > 1e-12:
        result = spearmanr(va, vb)
        rho = float(result[0])
    return {"distance_spearman": rho,
            "neighbor_overlap": float((np.minimum(wa, wb).sum(axis=1) / wa.sum(axis=1)).mean())}


def describe(values):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite metric.")
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p05": float(np.quantile(x, .05)), "p95": float(np.quantile(x, .95)),
            "images": len(x)}


def stratified_bootstrap(values, strata, seed=0, repetitions=1000):
    """Conditional CI: resample images within class/domain, not template rows.

    This does not estimate variation across training seeds or held-out domains.
    """
    values, strata = np.asarray(values), np.asarray(strata)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    estimates = np.zeros(repetitions)
    for group in groups:
        estimates += values[rng.choice(group, (repetitions, len(group)), replace=True)].mean(axis=1) / len(groups)
    return {"macro_class_domain_mean": float(np.mean([values[g].mean() for g in groups])),
            "image_bootstrap_95ci": np.quantile(estimates, [.025, .975]).tolist(),
            "bootstrap_repetitions": repetitions}


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def summarize(folder, neighbors=5):
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text())
    with (folder / "samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        samples = list(csv.DictReader(stream))
    labels = np.array([int(row["class_id"]) for row in samples])
    domains = np.array([row["domain"] for row in samples])
    e = np.load(folder / "e_unit.npy", mmap_mode="r")
    ln_e = np.load(folder / "ln_e_unit.npy", mmap_mode="r")
    constant = np.load(folder / "z_constant_unit.npy", mmap_mode="r")
    z = np.load(folder / "z_class_unit.npy", mmap_mode="r")
    z_mean = unit(z.mean(axis=1))
    scalars = dict(np.load(folder / "intervention_metrics.npz"))
    for array in [e, ln_e, constant, z, *scalars.values()]:
        if len(array) != len(samples):
            raise ValueError("Feature/manifest image count mismatch.")
    per_image = {
        "e_to_z_cosine_shift": cosine_shift(e[:, None], z).mean(axis=1),
        "ln_e_to_z_cosine_shift": cosine_shift(ln_e[:, None], z).mean(axis=1),
        "constant_to_z_cosine_shift": cosine_shift(constant[:, None], z).mean(axis=1),
        "template_pairwise_cosine_shift": template_spread(z),
        **scalars,
    }
    # The class word changes while template and image stay fixed. Compare with
    # changing the template while the image and true class word stay fixed.
    per_image["class_minus_template_shift"] = (per_image["class_swap_cosine_shift"] -
                                               per_image["template_pairwise_cosine_shift"])
    strata = np.array(["{}:{}".format(c, d) for c, d in zip(labels, domains)])
    paired = {name: {**describe(values), **stratified_bootstrap(values, strata, meta["sample_seed"])}
              for name, values in per_image.items()}
    image_rows = [{"image_id": row["image_id"], "class": row["class"], "domain": row["domain"],
                   **{name: float(values[i]) for name, values in per_image.items()}}
                  for i, row in enumerate(samples)]
    write_csv(folder / "per_image_metrics.csv", image_rows)
    representations = {"E": e, "LN(E)": ln_e, "Z_constant": constant, "Z_mean80": z_mean}
    metric_rows, topology_rows = [], []
    for c in sorted(set(labels.tolist())):
        mask = labels == c
        name = meta["class_names"][c]
        for representation, features in list(representations.items()) + [
                ("Z_template_{:02d}".format(k), z[:, k]) for k in range(z.shape[1])]:
            metric_rows.append({"class_id": c, "class": name, "representation": representation,
                                **domain_metrics(features[mask], domains[mask], neighbors)})
        for source, baseline in [("E", e), ("LN(E)", ln_e)]:
            for k in range(z.shape[1]):
                topology_rows.append({"class_id": c, "class": name, "source": source,
                                      "template_id": k, **topology_metrics(baseline[mask], z[mask, k], neighbors)})
    write_csv(folder / "class_domain_metrics.csv", metric_rows)
    write_csv(folder / "within_class_topology.csv", topology_rows)
    summary = {"protocol": meta, "neighbors": neighbors, "per_image": paired,
               "interpretation": [
                   "Positive within-class domain silhouette, cosine gap and neighbor excess support residual domain structure.",
                   "Small template spread does not imply that E and Z are identical; check both paired shifts and topology.",
                   "A positive class-minus-template contrast indicates larger class-word sensitivity under these swaps.",
                   "Ground-truth-conditioned Z is a mechanism diagnostic, not a label-free test representation.",
                   "Bootstrap intervals condition on one checkpoint and sampled domains; no training-seed uncertainty.",
                   "Performance benefit or equivalence requires separately trained B5B, Bconst and TDA-off controls.",
               ]}
    dump_json(folder / "summary.json", summary)
    lines = ["# CIPT mechanism diagnostics", "", "No conclusion is selected automatically.", "",
             "All quantitative metrics use unit features in the original embedding dimension.",
             "Each image contributes one value to intervention summaries. All 80 templates are evaluated.", "",
             "| Paired image metric | Mean | Image bootstrap 95% CI (class/domain macro) |",
             "|---|---:|---|" ]
    for name, stats in paired.items():
        lines.append("| {} | {:.6f} | {} |".format(name, stats["mean"], stats["image_bootstrap_95ci"]))
    lines += ["", "## Interpretation", ""] + ["- " + s for s in summary["interpretation"]]
    lines += ["", "## Files", "", "- class_domain_metrics.csv: E, LN(E), constant text, Z mean, and every fixed template; one row per class.",
              "- within_class_topology.csv: E/Z and LN(E)/Z neighborhood overlap and distance rank correlation, all templates.",
              "- per_image_metrics.csv: paired shifts, template spread, neutral-text and wrong-class swaps.",
              "- domain_metrics.png/pdf and intervention_sensitivity.png/pdf: quantitative summaries.",
              "- tsne/: one joint fit per class/seed/perplexity; arrows show correspondence only.",
              "", "Z_mean80 averages unit Z_k and normalizes again. It is a descriptive feature, not the legacy inference rule.",
              "Displayed domains include source/target designations from checkpoint metadata; dataset files may include training images."]
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return meta, samples, representations, z, metric_rows, per_image


def _plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    for suffix in [".png", ".pdf"]:
        fig.savefig(str(path) + suffix, dpi=160, bbox_inches="tight")


def plot_metrics(folder, rows, per_image):
    plt = _plot_setup()
    names = list(dict.fromkeys(row["class"] for row in rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, key, title in zip(axes, ["domain_silhouette", "neighbor_excess", "domain_cosine_gap"],
                             ["Within-class domain silhouette", "Same-domain neighbor excess", "Same-domain minus cross-domain cosine"]):
        for representation in ["E", "LN(E)", "Z_constant", "Z_mean80", "Z_all80_mean"]:
            values = []
            for name in names:
                selected = [r[key] for r in rows if r["class"] == name and
                            (r["representation"].startswith("Z_template_") if representation == "Z_all80_mean"
                             else r["representation"] == representation)]
                values.append(np.mean(selected))
            ax.plot(range(len(names)), values, marker="o", ms=3, label=representation)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="gray", lw=.7)
    axes[-1].legend(fontsize=8)
    _save(fig, Path(folder) / "domain_metrics")
    plt.close(fig)
    keys = ["e_to_z_cosine_shift", "ln_e_to_z_cosine_shift", "constant_to_z_cosine_shift",
            "template_pairwise_cosine_shift", "class_swap_cosine_shift", "neutral_swap_cosine_shift"]
    labels = ["E to Z", "LN(E) to Z", "Constant to Z", "Change template", "Change class word", "Class to subject"]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    label_key = "tick_labels" if "tick_labels" in inspect.signature(ax.boxplot).parameters else "labels"
    ax.boxplot([per_image[key] for key in keys], showfliers=False, **{label_key: labels})
    ax.set_ylabel("1 - cosine (one average per image)")
    ax.set_title("Paired intervention sensitivity; original embedding dimension")
    ax.tick_params(axis="x", labelrotation=15)
    _save(fig, Path(folder) / "intervention_sensitivity")
    plt.close(fig)


def joint_tsne(arrays, seed, perplexity, iterations=1000, pca_dim=50):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    x = np.concatenate([unit(a) for a in arrays])
    if not 0 < perplexity < len(x):
        raise ValueError("Perplexity must be >0 and less than the joint point count.")
    # Fit PCA once over all displayed representations, never separately.
    if pca_dim and min(x.shape) > pca_dim:
        x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    kwargs = dict(n_components=2, random_state=seed, perplexity=perplexity,
                  init="pca", learning_rate="auto")
    kwargs["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = iterations
    coords = TSNE(**kwargs).fit_transform(x)
    return np.split(coords, np.cumsum([len(a) for a in arrays])[:-1])


def plot_tsne(folder, meta, samples, representations, z, seeds, perplexities,
              template_id=0, iterations=1000, pca_dim=50):
    plt = _plot_setup()
    import sklearn
    from matplotlib.lines import Line2D
    labels = np.array([int(row["class_id"]) for row in samples])
    domains = np.array([row["domain"] for row in samples])
    domain_names = sorted(set(domains.tolist()))
    target_domains = set(meta.get("target_domains") or [])
    colors = plt.get_cmap("tab10")
    folder = Path(folder) / "tsne"
    folder.mkdir(exist_ok=True)
    keys = ["E", "LN(E)", "Z_constant", "Z_mean80", "Z_template_{:02d}".format(template_id)]
    arrays = list(representations.values()) + [z[:, template_id]]
    for seed in seeds:
        for perplexity in perplexities:
            for c in sorted(set(labels.tolist())):
                mask = labels == c
                coords = joint_tsne([a[mask] for a in arrays], seed, perplexity, iterations, pca_dim)
                basename = "class{}_k{}_seed{}_p{:g}".format(c, template_id, seed, perplexity)
                np.savez_compressed(folder / (basename + ".npz"),
                                    **dict(zip(keys, coords)), image_ids=np.flatnonzero(mask), domains=domains[mask])
                dump_json(folder / (basename + ".json"), {
                    "class_id": c, "template_id": template_id, "seed": seed,
                    "perplexity": perplexity, "iterations": iterations, "pca_dim_requested": pca_dim,
                    "pca_dim_effective": pca_dim if pca_dim and min(int(mask.sum()) * len(arrays), z.shape[-1]) > pca_dim else 0,
                    "sklearn_version": sklearn.__version__, "representations": keys,
                    "joint_points": int(mask.sum()) * len(arrays),
                    "checkpoint_sha256": meta.get("checkpoint_sha256"),
                    "samples_sha256": meta.get("samples_sha256"),
                })
                fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
                for ax, key, xy in zip(axes.flat, keys, coords):
                    for j, domain in enumerate(domain_names):
                        pts = xy[domains[mask] == domain]
                        ax.scatter(*pts.T, s=12, alpha=.7, color=colors(j))
                    ax.set_title(key)
                ax = axes.flat[-1]
                # Thin, deterministic subset of arrows: never interpret length.
                selected = np.linspace(0, len(coords[0]) - 1, min(20, len(coords[0])), dtype=int)
                for i in selected:
                    ax.annotate("", xy=coords[-1][i], xytext=coords[0][i],
                                arrowprops={"arrowstyle": "->", "color": "gray", "alpha": .25, "lw": .6})
                for j, domain in enumerate(domain_names):
                    dm = domains[mask] == domain
                    for points, marker in [(coords[0], "o"), (coords[-1], "x")]:
                        ax.scatter(*points[dm].T, marker=marker, color=colors(j), s=14, alpha=.6)
                ax.set_title("Paired E (o) to Z_k (x)")
                for ax in axes.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])
                legend = [Line2D([0], [0], color=colors(j), marker="o", linestyle="", label=domain +
                                  (" [role unknown]" if meta.get("target_domains") is None else
                                   " [target]" if domain in target_domains else " [source]"))
                          for j, domain in enumerate(domain_names)]
                fig.legend(handles=legend, loc="lower center", ncol=len(domain_names), fontsize=9)
                fig.suptitle("{}: joint t-SNE, seed={}, perplexity={:g}; arrows show pairing only".format(
                    meta["class_names"][c], seed, perplexity), fontsize=11)
                fig.tight_layout(rect=[0, .05, 1, .95])
                _save(fig, folder / basename)
                plt.close(fig)
                print("Rendered {}".format(basename), flush=True)
