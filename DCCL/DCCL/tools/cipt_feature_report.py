"""NumPy-only descriptive statistics and joint plots for paired CIPT texts.

All quantitative distances are measured in the original embedding dimension.
Images, rather than their 80 interventions, are units for class statistics.
"""

import csv
import inspect
import json
from pathlib import Path

import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def describe(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if not x.size:
        return None
    if not np.isfinite(x).all():
        raise ValueError("Non-finite feature/statistic encountered.")
    return {"mean": float(x.mean()), "std": float(x.std()),
            "median": float(np.median(x)), "min": float(x.min()),
            "max": float(x.max()), "count": int(x.size)}


def paired_metrics(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("Paired features must have identical shapes.")
    cosine = np.clip(np.sum(unit(a) * unit(b), axis=-1), -1, 1)
    return {"cosine": cosine, "cosine_shift": 1 - cosine,
            "l2_raw": np.linalg.norm(a - b, axis=-1),
            "l2_unit": np.linalg.norm(unit(a) - unit(b), axis=-1),
            "norm_a": np.linalg.norm(a, axis=-1),
            "norm_b": np.linalg.norm(b, axis=-1)}


def bank_metrics(bank):
    bank = np.asarray(bank)
    normalized = unit(bank)
    cosine = normalized @ normalized.T
    mask = np.triu(np.ones(cosine.shape, dtype=bool), 1)
    return {"norm": describe(np.linalg.norm(bank, axis=-1)),
            "pairwise_cosine_off_diagonal": describe(cosine[mask]),
            "centroid_norm": float(np.linalg.norm(bank.mean(axis=0)))}


def image_mean(z):
    """Mean of individually normalized z_k, normalized for geometric analysis."""
    return unit(unit(z).mean(axis=1))


def class_geometry(features, labels, class_names, domains=None):
    features, labels = unit(features), np.asarray(labels)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("Class statistics require one feature per image.")
    indices = sorted(set(labels.tolist()))
    centroids = np.stack([features[labels == c].mean(axis=0) for c in indices])
    centroid_unit = unit(centroids)
    cosine = np.clip(features @ features.T, -1, 1)
    upper = np.triu(np.ones(cosine.shape, dtype=bool), 1)
    same = labels[:, None] == labels[None, :]
    within, between = upper & same, upper & ~same
    within_d = (1 - cosine)[within]
    between_d = (1 - cosine)[between]
    result = {
        "unit_of_analysis": "one normalized feature per image (80-intervention mean for zk)",
        "classes": [class_names[c] for c in indices],
        "class_ids": indices,
        "images_per_class": [int((labels == c).sum()) for c in indices],
        "within_class_cosine": describe(cosine[within]),
        "between_class_cosine": describe(cosine[between]),
        "within_class_cosine_distance": describe(within_d),
        "between_class_cosine_distance": describe(between_d),
        "between_over_within_distance": (
            float(between_d.mean() / within_d.mean())
            if within_d.size and between_d.size and within_d.mean() > 1e-12
            else None),
        "centroid_cosine_distance": (1 - np.clip(
            centroid_unit @ centroid_unit.T, -1, 1)).tolist(),
        "centroid_l2_distance": np.linalg.norm(
            centroids[:, None] - centroids[None, :], axis=-1).tolist(),
    }
    if domains is not None:
        domains = np.asarray(domains)
        across = domains[:, None] != domains[None, :]
        result["same_class_cross_domain_cosine"] = describe(cosine[within & across])
        result["same_class_same_domain_cosine"] = describe(cosine[within & ~across])
    return result


def prompt_dispersion(z):
    normalized = unit(z)
    center = image_mean(z)
    return np.mean(1 - np.clip(
        np.sum(normalized * center[:, None, :], axis=-1), -1, 1), axis=1)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(features, class_names, selected_ids, labels=None, domains=None):
    p_c, p_a = features["pk_class"], features["pk_agnostic"]
    if p_c.shape[1] != 80 or p_a.shape[0] != 80:
        raise ValueError("This protocol requires the complete paired 80-text banks.")
    result = {"pk": {"class_conditioned": {},
                     "class_agnostic": bank_metrics(p_a),
                     "class_agnostic_between_class_comparison":
                     "Not applicable: one shared bank, no class label; do not duplicate it by class.",
                     "paired_by_class": {}}}
    for local, class_id in enumerate(selected_ids):
        name = class_names[class_id]
        result["pk"]["class_conditioned"][name] = bank_metrics(p_c[local])
        result["pk"]["paired_by_class"][name] = {
            k: describe(v) for k, v in paired_metrics(p_c[local], p_a).items()}
    centers = unit(p_c.mean(axis=1))
    result["pk"]["class_centroid_names"] = [class_names[i] for i in selected_ids]
    result["pk"]["class_centroid_cosine_distance"] = (
        1 - np.clip(centers @ centers.T, -1, 1)).tolist()
    result["pk"]["class_same_template_cosine"] = np.clip(
        np.einsum("ckd,ekd->ce", unit(p_c), unit(p_c)) / p_c.shape[1], -1, 1).tolist()
    result["norms"] = {k: describe(np.linalg.norm(v, axis=-1))
                       for k, v in features.items()
                       if k not in ("labels", "selected_ids")}
    if "zk_class" in features:
        zc, za = features["zk_class"], features["zk_agnostic"]
        if zc.shape != za.shape or zc.shape[0] != len(labels):
            raise ValueError("Paired interventions and image labels must match.")
        result["zk"] = {
            "paired_all_interventions": {
                k: describe(v) for k, v in paired_metrics(zc, za).items()},
            "paired_image_means": {k: describe(v) for k, v in
                                   paired_metrics(image_mean(zc), image_mean(za)).items()},
            "class_conditioned": class_geometry(
                image_mean(zc), labels, class_names, domains),
            "class_agnostic": class_geometry(
                image_mean(za), labels, class_names, domains),
            "E_baseline": class_geometry(features["e"], labels, class_names, domains),
            "prompt_dispersion_class": describe(prompt_dispersion(zc)),
            "prompt_dispersion_agnostic": describe(prompt_dispersion(za)),
        }
    return result


def joint_tsne(groups, seed=0, perplexity=30):
    """Fit once to all compared groups; never align separately fitted maps."""
    from sklearn.manifold import TSNE
    lengths = [len(group) for group in groups]
    combined = np.concatenate([unit(group) for group in groups]).astype(np.float32)
    if len(combined) < 4 or combined.shape[1] < 2:
        raise ValueError("t-SNE requires at least four points and two dimensions.")
    kwargs = dict(n_components=2, metric="cosine", init="pca",
                  perplexity=min(float(perplexity), max(2, (len(combined) - 1) / 3)),
                  random_state=seed, learning_rate=200.0, n_jobs=1)
    # scikit-learn renamed n_iter to max_iter in version 1.5.
    kwargs["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = 1000
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        coordinates = TSNE(**kwargs).fit_transform(combined)
    splits = np.cumsum(lengths)[:-1]
    return np.split(coordinates, splits), kwargs


def save_report(out, features, class_names, selected_ids, metadata,
                labels=None, domains=None, samples=None, make_tsne=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(features, class_names, selected_ids, labels, domains)
    summary["protocol"] = metadata
    np.savez_compressed(out / "features.npz", **features,
                        class_names=np.asarray(class_names),
                        selected_class_ids=np.asarray(selected_ids),
                        image_labels=np.asarray(labels if labels is not None else [], dtype=int),
                        image_domains=np.asarray(domains if domains is not None else []))
    rows = []
    for local, c in enumerate(selected_ids):
        metrics = paired_metrics(features["pk_class"][local], features["pk_agnostic"])
        for k in range(80):
            rows.append({"class": class_names[c], "template_id": k,
                         **{name: float(values[k]) for name, values in metrics.items()},
                         "norm_raw_class": float(np.linalg.norm(features["pk_raw_class"][local, k])),
                         "norm_raw_agnostic": float(np.linalg.norm(features["pk_raw_agnostic"][k]))})
    write_csv(out / "pk_paired.csv", rows)
    if samples is not None:
        write_csv(out / "samples.csv", samples)
    if "zk_class" in features:
        zc, za = features["zk_class"], features["zk_agnostic"]
        metrics = paired_metrics(zc, za)
        e = unit(features["e"])[:, None, :]
        p_to_v_c = np.sum(features["visual"][:, None] * features["pk_class"][
            [selected_ids.index(int(c)) for c in labels]], axis=-1)
        p_to_v_a = features["visual"] @ features["pk_agnostic"].T
        rows, mean_rows = [], []
        mean_pairs = paired_metrics(image_mean(zc), image_mean(za))
        for i, c in enumerate(labels):
            prefix = {"image_id": i, "class": class_names[c], "domain": domains[i]}
            for k in range(80):
                rows.append({**prefix, "template_id": k,
                    **{name: float(values[i, k]) for name, values in metrics.items()},
                    "cosine_z_E_class": float(np.sum(unit(zc[i, k]) * e[i, 0])),
                    "cosine_z_E_agnostic": float(np.sum(unit(za[i, k]) * e[i, 0])),
                    "cosine_pk_visual_class": float(p_to_v_c[i, k]),
                    "cosine_pk_visual_agnostic": float(p_to_v_a[i, k]),
                    "norm_preLN_class": float(np.linalg.norm(features["zk_preln_class"][i, k])),
                    "norm_preLN_agnostic": float(np.linalg.norm(features["zk_preln_agnostic"][i, k]))})
            mean_rows.append({**prefix,
                **{name: float(values[i]) for name, values in mean_pairs.items()},
                "prompt_dispersion_class": float(prompt_dispersion(zc[i:i+1])[0]),
                "prompt_dispersion_agnostic": float(prompt_dispersion(za[i:i+1])[0])})
        write_csv(out / "zk_paired.csv", rows)
        write_csv(out / "zk_image_means.csv", mean_rows)
    make_plots(out, features, summary, class_names, selected_ids, metadata,
               labels, domains, make_tsne)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    protocol_line = ("Frozen text-encoder pilot, using all 80 paired texts." if metadata.get("pk_only")
                     else "Descriptive pilot, using a fixed checkpoint and all 80 paired texts.")
    lines = ["# CIPT paired text feature diagnostics", "", protocol_line,
             "Quantitative metrics use the original feature dimension.",
             "Class statistics use one feature per image; 80 texts are not 80 independent images.", "",
             "| Class | Mean pk cosine | Mean pk cosine shift |", "|---|---:|---:|"]
    for name, stats in summary["pk"]["paired_by_class"].items():
        lines.append("| {} | {:.6f} | {:.6f} |".format(
            name, stats["cosine"]["mean"], stats["cosine_shift"]["mean"]))
    if "zk" in summary:
        lines += ["", "| Image-level metric | B5B class text | B5A subject text |",
                  "|---|---:|---:|"]
        for key in ["within_class_cosine", "between_class_cosine", "same_class_cross_domain_cosine"]:
            values = [summary["zk"][condition][key] for condition in
                      ["class_conditioned", "class_agnostic"]]
            formatted = ["N/A" if v is None else "{:.6f}".format(v["mean"]) for v in values]
            lines.append("| {} | {} | {} |".format(key, *formatted))
        lines += ["", "Mean paired image-level z cosine shift: {:.6f}".format(
            summary["zk"]["paired_image_means"]["cosine_shift"]["mean"])]
    lines += ["", "Unit pk norms are fixed by normalization; z norms are constrained by LayerNorm.",
              "A shared subject bank has no class label: zero between-class pk distance is a construction property.",
              "t-SNE map distances and cluster sizes are not quantitative evidence of feature separation.",
              "This paired intervention test measures text replacement in one model, not performance of two trained models."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def make_plots(out, features, summary, class_names, selected_ids, metadata,
               labels, domains, make_tsne):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = plt.get_cmap("tab10")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})

    def save(fig, name):
        if metadata.get("synthetic_validation_only"):
            fig.text(.5, .01, "SYNTHETIC DATA - rendering validation only", ha="center", color="#993333", fontsize=9)
        fig.savefig(out / (name + ".png"), bbox_inches="tight")
        fig.savefig(out / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    def heatmaps(matrices, names, titles, filename):
        fig, axes = plt.subplots(1, len(matrices), figsize=(4 * len(matrices), 3.6), squeeze=False)
        vmax = max(0.05, max(float(np.max(m)) for m in matrices))
        for ax, matrix, title in zip(axes[0], matrices, titles):
            matrix = np.asarray(matrix)
            im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
            ax.set_yticks(range(len(names)), names)
            ax.set_title(title)
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, "{:.3f}".format(matrix[i, j]), ha="center", va="center",
                            color="white" if matrix[i, j] > vmax * .6 else "black", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
        save(fig, filename)

    names = [class_names[i] for i in selected_ids]
    heatmaps([summary["pk"]["class_centroid_cosine_distance"]], names,
             ["pk: B5B class centroid cosine distance"], "pk_class_distance")
    norm_groups = [("Text encoder raw", "pk_raw_class", "pk_raw_agnostic"),
                   ("pk: unit text", "pk_class", "pk_agnostic")]
    if "zk_class" in features:
        norm_groups += [("z before LayerNorm", "zk_preln_class", "zk_preln_agnostic"),
                        ("zk after LayerNorm", "zk_class", "zk_agnostic")]
        groups = ["E_baseline", "class_conditioned", "class_agnostic"]
        heatmaps([summary["zk"][g]["centroid_cosine_distance"] for g in groups],
                 summary["zk"]["class_conditioned"]["classes"],
                 ["E baseline", "zk image mean: B5B", "zk image mean: B5A"], "zk_class_distance")
    fig, axes = plt.subplots(1, len(norm_groups), figsize=(3.4 * len(norm_groups), 3.3), squeeze=False)
    for ax, (title, key_c, key_a) in zip(axes[0], norm_groups):
        nc, na = np.linalg.norm(features[key_c], axis=-1).ravel(), np.linalg.norm(features[key_a], axis=-1).ravel()
        label_key = "tick_labels" if "tick_labels" in inspect.signature(ax.boxplot).parameters else "labels"
        ax.boxplot([nc, na], **{label_key: ["B5B", "B5A"]})
        ax.set_title(title)
        ax.set_ylabel("L2 norm")
    save(fig, "feature_norms")
    if not make_tsne:
        return

    seed, perplexity = metadata["seed"], metadata["tsne_perplexity"]
    pc = features["pk_class"].reshape(-1, features["pk_class"].shape[-1])
    pa = features["pk_agnostic"]
    coords, settings = joint_tsne([pc, pa], seed, perplexity)
    metadata["pk_tsne"] = {k: v for k, v in settings.items() if isinstance(v, (int, float, str))}
    np.savez_compressed(out / "pk_tsne_coordinates.npz", class_text=coords[0], subject_text=coords[1])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for local, name in enumerate(names):
        points = coords[0][local * 80:(local + 1) * 80]
        ax.scatter(*points.T, s=14, color=colors(local), label="B5B " + name, alpha=.7)
    ax.scatter(*coords[1].T, s=23, marker="x", color="#343434", label="B5A subject")
    ax.set_title("pk: all 80 texts per bank, one joint t-SNE fit")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=9)
    save(fig, "pk_joint_tsne")
    if "zk_class" not in features:
        return

    zc, za = features["zk_class"], features["zk_agnostic"]
    # Select the SAME template ids for every paired image if an explicit plotting cap is requested.
    count = min(80, metadata["tsne_templates_per_image"])
    template_ids = np.sort(np.random.default_rng(seed).choice(80, count, replace=False))
    points_c = zc[:, template_ids].reshape(-1, zc.shape[-1])
    points_a = za[:, template_ids].reshape(-1, za.shape[-1])
    coords, settings = joint_tsne([points_c, points_a], seed, perplexity)
    metadata["zk_tsne"] = {k: v for k, v in settings.items() if isinstance(v, (int, float, str))}
    metadata["zk_tsne_template_ids"] = template_ids.tolist()
    np.savez_compressed(out / "zk_tsne_coordinates.npz", class_text=coords[0], subject_text=coords[1],
                        template_ids=template_ids, image_labels=np.asarray(labels))
    repeated_labels = np.repeat(labels, count)
    domain_names = sorted(set(np.asarray(domains).tolist()))
    domain_markers = ["o", "^", "s", "D", "v", "P"]
    repeated_domains = np.repeat(domains, count)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, points, title in zip(axes, coords, ["B5B class text", "B5A subject text"]):
        for local, c in enumerate(selected_ids):
            for domain_id, domain in enumerate(domain_names):
                p = points[(repeated_labels == c) & (repeated_domains == domain)]
                ax.scatter(*p.T, color=colors(local), s=9, alpha=.45,
                           marker=domain_markers[domain_id % len(domain_markers)])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    legend = [Line2D([0], [0], marker="o", color=colors(i), linestyle="", label=n) for i, n in enumerate(names)]
    legend += [Line2D([0], [0], marker=domain_markers[i % len(domain_markers)], color="gray", linestyle="", label=n)
               for i, n in enumerate(domain_names)]
    axes[1].legend(handles=legend, fontsize=9)
    fig.suptitle("zk: one joint fit, {} paired templates per image".format(count))
    save(fig, "zk_joint_tsne")
    # Per-image means avoid mistaking 80 correlated interventions for 80 images.
    means = [image_mean(zc), image_mean(za)]
    coords, settings = joint_tsne(means, seed, min(10, perplexity))
    np.savez_compressed(out / "zk_mean_tsne_coordinates.npz", class_text=coords[0], subject_text=coords[1])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, c in enumerate(labels):
        local = selected_ids.index(int(c))
        ax.plot([coords[0][i, 0], coords[1][i, 0]], [coords[0][i, 1], coords[1][i, 1]],
                color=colors(local), alpha=.2, linewidth=.7)
        for point, marker in zip([coords[0][i], coords[1][i]], ["o", "x"]):
            ax.scatter(*point, color=colors(local), marker=marker, s=33)
    legend = [Line2D([0], [0], marker="o", color=colors(i), linestyle="", label=n)
              for i, n in enumerate(names)]
    legend += [Line2D([0], [0], marker=m, color="gray", linestyle="", label=n)
               for m, n in [("o", "B5B"), ("x", "B5A")]]
    ax.legend(handles=legend, fontsize=9)
    ax.set_title("zk image means: paired points, one joint t-SNE fit")
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "zk_image_mean_joint_tsne")
