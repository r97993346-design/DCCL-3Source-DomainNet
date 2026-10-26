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


def native_z_info(meta):
    """Old archives are B5B; new archives explicitly identify their native Z."""
    mode = str(meta.get("template_mode", meta.get("checkpoint_hparams", {}).get(
        "cipt_template_mode", "b5b"))).lower()
    if mode not in {"b5b", "b5a"}:
        raise ValueError("Reports support only B5B or paired B5A native Z.")
    subject = meta.get("neutral_subject", meta.get("checkpoint_hparams", {}).get("cipt_neutral_subject", "subject"))
    condition = ("B5B: true-class texts; mechanism diagnostic" if mode == "b5b" else
                 "B5A: shared '{}' texts; no label-conditioned fusion".format(subject))
    return {"template_mode": mode, "z_uses_ground_truth_class_word": mode == "b5b",
            "z_condition": condition}


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
    native = native_z_info(meta)
    default_z_file = "z_class_unit.npy" if native["template_mode"] == "b5b" else "z_neutral_unit.npy"
    z = np.load(folder / meta.get("z_file", default_z_file), mmap_mode="r")
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
    # B5A contains no class noun. Do not fabricate a zero class-swap score or
    # subtract its native template spread from an unrelated class-word probe.
    if native["template_mode"] == "b5b":
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
    interpretation = [
        "Positive within-class domain silhouette, cosine gap and neighbor excess support residual domain structure.",
        "Small template spread does not imply that E and Z are identical; check both paired shifts and topology.",
        "Bootstrap intervals condition on one checkpoint and sampled domains; no training-seed uncertainty.",
        "Performance benefit or equivalence requires separately trained B5B, Bconst and TDA-off controls.",
        "Z_mean80 summarizes geometry; it is not the model's K-branch logit aggregation rule.",
    ]
    if native["template_mode"] == "b5b":
        interpretation += [
            "A positive class-minus-template contrast indicates larger class-word sensitivity under these swaps.",
            "Ground-truth-conditioned Z is a mechanism diagnostic, not a label-free test representation.",
        ]
    else:
        interpretation += [
            "Native B5A Z uses the saved shared neutral-subject bank; image labels only group the domain statistics.",
            "Class insertion is a post-training text probe, not a native class swap or evidence of training benefit.",
            "Compare separately trained B5B/B5A E on the same manifest to study learned visual representations.",
        ]
    summary = {"protocol": meta, "native_z": native, "neighbors": neighbors,
               "per_image": paired, "interpretation": interpretation}
    dump_json(folder / "summary.json", summary)
    lines = ["# CIPT mechanism diagnostics", "", native["z_condition"], "",
             "No conclusion is selected automatically.", "",
             "All quantitative metrics use unit features in the original embedding dimension.",
             "Each image contributes one value to intervention summaries. All 80 templates are evaluated.", "",
             "| Paired image metric | Mean | Image bootstrap 95% CI (class/domain macro) |",
             "|---|---:|---|" ]
    for name, stats in paired.items():
        lines.append("| {} | {:.6f} | {} |".format(name, stats["mean"], stats["image_bootstrap_95ci"]))
    lines += ["", "## Interpretation", ""] + ["- " + s for s in summary["interpretation"]]
    lines += ["", "## Files", "", "- class_domain_metrics.csv: E, LN(E), constant text, Z mean, and every fixed template; one row per class.",
              "- within_class_topology.csv: E/Z and LN(E)/Z neighborhood overlap and distance rank correlation, all templates.",
              "- per_image_metrics.csv: native E/Z shifts, template spread, and explicitly named post-training probes.",
              "- domain_metrics.png/pdf and intervention_sensitivity.png/pdf: quantitative summaries.",
              "- tsne/: one joint fit per class/seed/perplexity; arrows show correspondence only.",
              "- tsne_templates/ (unless skipped): fixed images and native subject words, varying templates; one joint fit per class.",
              "- tsne_overview/ (with --overview): all selected classes; colors = classes, markers = domains.",
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


def plot_metrics(folder, rows, per_image, meta=None):
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
    options = [("e_to_z_cosine_shift", "E to native Z"), ("ln_e_to_z_cosine_shift", "LN(E) to native Z"),
               ("constant_to_z_cosine_shift", "Constant to native Z"),
               ("template_pairwise_cosine_shift", "Change template"),
               ("class_swap_cosine_shift", "Class swap (probe)"),
               ("neutral_swap_cosine_shift", "Class to neutral (probe)"),
               ("class_insertion_cosine_shift", "Neutral to class (probe)")]
    keys, labels = zip(*[(key, label) for key, label in options if key in per_image])
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    label_key = "tick_labels" if "tick_labels" in inspect.signature(ax.boxplot).parameters else "labels"
    ax.boxplot([per_image[key] for key in keys], showfliers=False, **{label_key: labels})
    ax.set_ylabel("1 - cosine (one average per image)")
    ax.set_title("Native Z and post-training text probes; original embedding dimension" +
                 ("\n" + native_z_info(meta)["z_condition"] if meta is not None else ""))
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
                _archive_tsne(folder, basename, keys, coords, meta, samples, np.flatnonzero(mask),
                              seed, perplexity, iterations, pca_dim, z.shape[-1],
                              class_id=c, template_id=template_id)
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
                fig.suptitle("{}: joint t-SNE, seed={}, perplexity={:g}; arrows show pairing only\n{}".format(
                    meta["class_names"][c], seed, perplexity, native_z_info(meta)["z_condition"]), fontsize=11)
                fig.tight_layout(rect=[0, .05, 1, .92])
                _save(fig, folder / basename)
                plt.close(fig)
                print("Rendered {}".format(basename), flush=True)


def _archive_tsne(folder, basename, keys, coords, meta, samples, indices,
                  seed, perplexity, iterations, pca_dim, input_dim, **settings):
    """Persist exact sample pairing, coordinates and each joint fit's settings."""
    import sklearn
    selected = [samples[i] for i in indices]
    point_count = sum(len(xy) for xy in coords)
    np.savez_compressed(folder / (basename + ".npz"), **dict(zip(keys, coords)),
                        sample_indices=indices,
                        image_ids=np.array([str(row["image_id"]) for row in selected]),
                        domains=np.array([row["domain"] for row in selected]),
                        class_ids=np.array([int(row["class_id"]) for row in selected]))
    dump_json(folder / (basename + ".json"), {
        "seed": seed, "perplexity": perplexity, "iterations": iterations,
        "pca_dim_requested": pca_dim,
        "pca_dim_effective": pca_dim if pca_dim and min(point_count, input_dim) > pca_dim else 0,
        "sklearn_version": sklearn.__version__, "representations": keys,
        "joint_points": point_count, "unique_images": len(selected),
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "samples_sha256": meta.get("samples_sha256"),
        "samples_identity_sha256": meta.get("samples_identity_sha256"),
        **native_z_info(meta),
        "coordinate_scope": "Only panels in this file share a joint fit. Distances are illustrative.",
        **settings,
    })


def _domain_label(domain, meta):
    targets = meta.get("target_domains")
    role = "role unknown" if targets is None else "target" if domain in targets else "source"
    return "{} [{}]".format(domain, role)


def plot_template_tsne(folder, meta, samples, representations, z, seeds, perplexities,
                       template_ids=(0, 20, 40, 60), iterations=1000, pca_dim=50):
    """Same image/native subject word, different text descriptions.

    All images enter every panel. The last panel connects a deterministic,
    domain-balanced subset for legibility; it never selects by measured shift.
    """
    template_ids = list(template_ids)
    if (len(template_ids) < 2 or len(set(template_ids)) != len(template_ids)
            or any(not 0 <= k < z.shape[1] for k in template_ids)):
        raise ValueError("Need >=2 distinct, in-range template IDs.")
    plt = _plot_setup()
    from matplotlib.lines import Line2D
    folder = Path(folder) / "tsne_templates"
    folder.mkdir(exist_ok=True)
    labels = np.array([int(row["class_id"]) for row in samples])
    domains = np.array([row["domain"] for row in samples])
    domain_names = sorted(set(domains.tolist()))
    colors = plt.get_cmap("tab10")
    keys = ["Z_constant"] + ["Z_template_{:02d}".format(k) for k in template_ids]
    arrays = [representations["Z_constant"]] + [z[:, k] for k in template_ids]
    tag = "-".join(str(k) for k in template_ids)
    for seed in seeds:
        for perplexity in perplexities:
            for c in sorted(set(labels.tolist())):
                indices = np.flatnonzero(labels == c)
                local_domains = domains[indices]
                paired = []
                for domain in domain_names:
                    group = np.flatnonzero(local_domains == domain)
                    paired.extend(group[np.linspace(0, len(group) - 1, min(2, len(group)), dtype=int)])
                coords = joint_tsne([a[indices] for a in arrays], seed, perplexity, iterations, pca_dim)
                basename = "class{}_templates{}_seed{}_p{:g}".format(c, tag, seed, perplexity)
                _archive_tsne(folder, basename, keys, coords, meta, samples, indices,
                              seed, perplexity, iterations, pca_dim, z.shape[-1],
                              class_id=c, template_ids=template_ids,
                              paired_sample_indices=[int(indices[i]) for i in paired],
                              pairing_rule="Up to two images per domain, evenly spaced in manifest order; no shift-based selection.",
                              intervention=("Image and native subject word fixed; only text description varies. " +
                                            native_z_info(meta)["z_condition"]))
                panel_count = len(keys) + 1
                ncols = min(3, panel_count)
                nrows = (panel_count + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows),
                                         sharex=True, sharey=True, squeeze=False)
                for ax, key, xy in zip(axes.flat, keys, coords):
                    for j, domain in enumerate(domain_names):
                        ax.scatter(*xy[local_domains == domain].T, s=12, alpha=.65, color=colors(j))
                    ax.set_title(key, fontsize=10)
                ax = axes.flat[len(keys)]
                # No temporal ordering is implied: lines connect conditions.
                for i in paired:
                    xy = np.array([points[i] for points in coords[1:]])
                    color = colors(domain_names.index(local_domains[i]))
                    ax.plot(*xy.T, color=color, alpha=.55, lw=.8)
                    ax.scatter(*xy.T, color=color, s=18)
                    ax.annotate(str(samples[indices[i]]["image_id"]), xy[0], fontsize=6,
                                xytext=(3, 3), textcoords="offset points")
                ax.set_title("Same image across templates (subset)", fontsize=10)
                # Pairing panel uses the same limits as every full-sample panel.
                for ax in list(axes.flat)[:panel_count]:
                    ax.set_xticks([])
                    ax.set_yticks([])
                for ax in list(axes.flat)[panel_count:]:
                    ax.set_visible(False)
                legend = [Line2D([0], [0], color=colors(j), marker="o", linestyle="",
                                 label=_domain_label(domain, meta)) for j, domain in enumerate(domain_names)]
                fig.legend(handles=legend, loc="lower center", ncol=len(domain_names), fontsize=8)
                fig.suptitle("{}: fixed images/native subject words, varying descriptions\n{}\n"
                             "joint t-SNE; lines indicate correspondence, not feature-distance magnitude".format(
                                 meta["class_names"][c], native_z_info(meta)["z_condition"]), fontsize=11)
                fig.tight_layout(rect=[0, .05, 1, .88])
                _save(fig, folder / basename)
                plt.close(fig)
                print("Rendered {}".format(basename), flush=True)


def plot_overview_tsne(folder, meta, samples, representations, z, seeds, perplexities,
                       template_id=0, iterations=1000, pca_dim=50):
    """Optional class overview; domain-cluster evidence remains class-conditional."""
    plt = _plot_setup()
    from matplotlib.lines import Line2D
    labels = np.array([int(row["class_id"]) for row in samples])
    domains = np.array([row["domain"] for row in samples])
    classes = sorted(set(labels.tolist()))
    domain_names = sorted(set(domains.tolist()))
    if len(classes) > 20 or len(domain_names) > 8:
        raise ValueError("Overview supports <=20 selected classes and <=8 domains; choose a documented subset.")
    colors = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")
    markers = ["o", "^", "s", "X", "D", "v", "P", "*"]
    folder = Path(folder) / "tsne_overview"
    folder.mkdir(exist_ok=True)
    keys = ["E", "Z_constant", "Z_template_{:02d}".format(template_id)]
    arrays = [representations["E"], representations["Z_constant"], z[:, template_id]]
    indices = np.arange(len(samples))
    for seed in seeds:
        for perplexity in perplexities:
            coords = joint_tsne(arrays, seed, perplexity, iterations, pca_dim)
            basename = "all_classes_k{}_seed{}_p{:g}".format(template_id, seed, perplexity)
            _archive_tsne(folder, basename, keys, coords, meta, samples, indices,
                          seed, perplexity, iterations, pca_dim, z.shape[-1],
                          template_id=template_id, colors="classes", markers="domains",
                          class_color_order=classes, domain_marker_order=domain_names)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True)
            for ax, key, xy in zip(axes, keys, coords):
                for j, c in enumerate(classes):
                    for k, domain in enumerate(domain_names):
                        mask = (labels == c) & (domains == domain)
                        ax.scatter(*xy[mask].T, s=15, alpha=.6, color=colors(j),
                                   marker=markers[k], linewidths=.3)
                ax.set_title(key)
                ax.set_xticks([])
                ax.set_yticks([])
            class_legend = [Line2D([0], [0], color=colors(j), marker="o", linestyle="",
                                   label=meta["class_names"][c]) for j, c in enumerate(classes)]
            domain_legend = [Line2D([0], [0], color="gray", marker=markers[j], linestyle="",
                                    label=_domain_label(domain, meta)) for j, domain in enumerate(domain_names)]
            class_rows = (len(classes) + 6) // 7
            legend_space = .09 + .04 * class_rows
            fig.legend(handles=class_legend, loc="lower center", ncol=min(7, len(classes)),
                       bbox_to_anchor=(.5, .05), fontsize=8)
            fig.legend(handles=domain_legend, loc="lower center", ncol=len(domain_names),
                       bbox_to_anchor=(.5, .005), fontsize=8)
            fig.suptitle("Joint t-SNE: class colors, domain markers; seed={}, perplexity={:g}\n"
                         "{}".format(seed, perplexity, native_z_info(meta)["z_condition"]), fontsize=11)
            fig.tight_layout(rect=[0, legend_space, 1, .90])
            _save(fig, folder / basename)
            plt.close(fig)
            print("Rendered {}".format(basename), flush=True)

