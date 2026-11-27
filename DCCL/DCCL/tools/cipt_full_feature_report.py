"""Full-dataset CIPT reports: streamed interventions and exact mean pair metrics.

Image pairs are summarized using vector sums, without constructing N-by-N
similarity matrices. All numerical metrics stay in the original feature space.
"""

import argparse
import csv
import inspect
import json
from pathlib import Path

import numpy as np

from cipt_feature_report import (image_mean, paired_metrics, prompt_dispersion,
                                 save_report, summarize, unit, write_csv)


class OnlineStats:
    def __init__(self):
        self.count = 0
        self.total = self.squared = 0.0
        self.minimum, self.maximum = float("inf"), float("-inf")

    def add(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite feature statistic.")
        if values.size:
            self.count += int(values.size)
            self.total += float(values.sum())
            self.squared += float(values @ values)
            self.minimum = min(self.minimum, float(values.min()))
            self.maximum = max(self.maximum, float(values.max()))

    def result(self):
        if not self.count:
            return None
        mean = self.total / self.count
        return {"mean": mean, "std": float(np.sqrt(max(0, self.squared / self.count - mean * mean))),
                "min": self.minimum, "max": self.maximum, "count": self.count}


def mean_stat(total, count):
    return {"mean": float(np.clip(total / count, -1, 1)), "count": int(count)} if count else None


def exact_class_geometry(features, labels, class_names, domains):
    """Exact means over ALL image pairs, using sum_i u_i and sum_i ||u_i||².

    For one group: sum_{i<j} u_i·u_j = (||sum_i u_i||²-sum_i ||u_i||²)/2.
    For two disjoint groups: sum_{i,j} u_i·v_j = (sum_i u_i)·(sum_j v_j).
    """
    features = unit(features)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite image feature.")
    labels, domains = np.asarray(labels), np.asarray(domains)
    ids, domain_names = sorted(set(labels.tolist())), sorted(set(domains.tolist()))
    if features.ndim != 2 or len(features) != len(labels) or len(labels) != len(domains):
        raise ValueError("One feature, label and domain are required per image.")
    groups = {}
    for c in ids:
        for domain in domain_names:
            values = features[(labels == c) & (domains == domain)]
            groups[c, domain] = (len(values), values.sum(axis=0), float(np.sum(values * values)))
    class_sums = {c: features[labels == c].sum(axis=0) for c in ids}
    class_counts = {c: int((labels == c).sum()) for c in ids}
    within_total = cross_total = same_domain_total = 0.0
    within_count = cross_count = same_domain_count = 0
    per_class = []
    for c in ids:
        n, vector_sum = class_counts[c], class_sums[c]
        q = float(np.sum(features[labels == c] ** 2))
        pairs = n * (n - 1) // 2
        total = (float(vector_sum @ vector_sum) - q) / 2
        within_total += total
        within_count += pairs
        class_cross_total, class_cross_count = 0.0, 0
        for i, a in enumerate(domain_names):
            na, sa, qa = groups[c, a]
            same_domain_total += (float(sa @ sa) - qa) / 2
            same_domain_count += na * (na - 1) // 2
            for b in domain_names[i + 1:]:
                nb, sb, _ = groups[c, b]
                class_cross_total += float(sa @ sb)
                class_cross_count += na * nb
        cross_total += class_cross_total
        cross_count += class_cross_count
        per_class.append({"class_id": c, "class": class_names[c], "image_count": n,
                          "within_class_cosine": mean_stat(total, pairs),
                          "same_class_cross_domain_cosine": mean_stat(class_cross_total, class_cross_count)})
    between_total, between_count, class_pairs = 0.0, 0, []
    for i, c in enumerate(ids):
        for other in ids[i + 1:]:
            pair_total = float(class_sums[c] @ class_sums[other])
            pair_count = class_counts[c] * class_counts[other]
            between_total += pair_total
            between_count += pair_count
            class_pairs.append({"class_a": class_names[c], "class_b": class_names[other],
                                "pair_count": pair_count, "mean_cosine": pair_total / pair_count,
                                "mean_cosine_distance": 1 - pair_total / pair_count})
    centroids = np.stack([class_sums[c] / class_counts[c] for c in ids])
    centroids_unit = unit(centroids)
    centroid_cosine_distance = np.clip(1 - centroids_unit @ centroids_unit.T, 0, 2)
    within = mean_stat(within_total, within_count)
    between = mean_stat(between_total, between_count)
    result = {"unit_of_analysis": "one normalized original-dimensional feature per image",
              "pair_summary": "exact means over all pairs; no pair sampling; standard deviations/medians not materialized",
              "classes": [class_names[c] for c in ids], "class_ids": ids,
              "images_per_class": [class_counts[c] for c in ids],
              "within_class_cosine": within, "between_class_cosine": between,
              "within_class_cosine_distance": ({"mean": 1 - within["mean"], "count": within_count} if within else None),
              "between_class_cosine_distance": ({"mean": 1 - between["mean"], "count": between_count} if between else None),
              "between_over_within_distance": ((1 - between["mean"]) / (1 - within["mean"])
                  if within and between and 1 - within["mean"] > 1e-12 else None),
              "same_class_cross_domain_cosine": mean_stat(cross_total, cross_count),
              "same_class_same_domain_cosine": mean_stat(same_domain_total, same_domain_count),
              "centroid_cosine_distance": centroid_cosine_distance.tolist(),
              "centroid_l2_distance": np.linalg.norm(centroids[:, None] - centroids[None], axis=-1).tolist(),
              "centroid_unit_l2_distance": np.linalg.norm(centroids_unit[:, None] - centroids_unit[None], axis=-1).tolist(),
              "per_class": per_class, "class_pairs": class_pairs,
              "zero_norm_image_features": int((np.linalg.norm(features, axis=-1) < 1e-12).sum()),
              "macro_within_class_cosine": (float(np.mean([r["within_class_cosine"]["mean"] for r in per_class if r["within_class_cosine"]])) if within_count else None),
              "macro_between_class_cosine": (float(np.mean([r["mean_cosine"] for r in class_pairs])) if class_pairs else None),
              "macro_same_class_cross_domain_cosine": (float(np.mean([
                  r["same_class_cross_domain_cosine"]["mean"] for r in per_class if r["same_class_cross_domain_cosine"]])) if cross_count else None)}
    domain_matrix, domain_counts = np.zeros((len(domain_names), len(domain_names))), np.zeros((len(domain_names), len(domain_names)), dtype=np.int64)
    for i, a in enumerate(domain_names):
        for j, b in enumerate(domain_names):
            total, count = 0.0, 0
            for c in ids:
                na, sa, qa = groups[c, a]
                nb, sb, _ = groups[c, b]
                if i == j:
                    total += (float(sa @ sa) - qa) / 2
                    count += na * (na - 1) // 2
                else:
                    total += float(sa @ sb)
                    count += na * nb
            domain_matrix[i, j] = total / count if count else 0
            domain_counts[i, j] = count
    result["domain_names"] = domain_names
    result["same_class_domain_cosine_matrix"] = domain_matrix.tolist()
    result["same_class_domain_pair_counts"] = domain_counts.tolist()
    return result


class FullFeatureReport:
    def __init__(self, out, samples, class_names, selected_ids, pk, metadata, save_all_z=False):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        if any((self.out / name).exists() for name in ["summary.json", "samples.csv", "zk_paired.csv"]):
            raise FileExistsError("Use a fresh --output-dir for full extraction; use tools/cipt_full_feature_report.py to redraw existing results.")
        self.samples, self.class_names, self.ids, self.pk, self.metadata = samples, class_names, selected_ids, pk, metadata
        self.labels = np.asarray([s["class_id"] for s in samples], dtype=int)
        self.domains = np.asarray([s["domain"] for s in samples])
        self.written = 0
        self.norms, self.paired, self.mean_paired = {}, {}, {}
        self.dispersion = {"class": OnlineStats(), "agnostic": OnlineStats()}
        self.paired_by_class = {c: OnlineStats() for c in selected_ids}
        self.paired_by_domain = {d: OnlineStats() for d in sorted(set(self.domains.tolist()))}
        n, dim = len(samples), pk["pk_class"].shape[-1]
        vectors = self.out / "vectors"
        vectors.mkdir()
        self.vectors = {key: np.lib.format.open_memmap(vectors / (key + ".npy"), mode="w+", dtype=np.float32, shape=(n, dim))
                        for key in ["visual", "e", "zk_image_mean_class", "zk_image_mean_agnostic"]}
        self.all_z = ({key: np.lib.format.open_memmap(vectors / (key + ".npy"), mode="w+", dtype=np.float32, shape=(n, 80, dim))
                       for key in ["zk_class", "zk_agnostic"]} if save_all_z else {})
        metadata["save_all_z"] = save_all_z
        metadata["image_count"] = n
        metadata["interventions_per_condition"] = n * 80
        metadata["sample_counts"] = [{"class": class_names[c], "domain": d,
            "count": int(((self.labels == c) & (self.domains == d)).sum())}
            for d in sorted(set(self.domains.tolist())) for c in selected_ids]
        write_csv(self.out / "samples.csv", samples)
        np.savez_compressed(self.out / "pk_features.npz", **pk)
        self.handles = {}
        self.writers = {}
        print("Full coverage: {} images, {} classes, {} domains; {} interventions per condition".format(
            n, len(selected_ids), len(set(self.domains.tolist())), n * 80), flush=True)
        if save_all_z:
            print("Disk space for raw zk arrays: {:.2f} GiB".format(2 * n * 80 * dim * 4 / 2**30), flush=True)

    def rows(self, name, rows):
        if not rows:
            return
        if name not in self.writers:
            handle = (self.out / name).open("w", encoding="utf-8-sig", newline="")
            self.handles[name] = handle
            self.writers[name] = csv.DictWriter(handle, fieldnames=list(rows[0]))
            self.writers[name].writeheader()
        self.writers[name].writerows(rows)

    @staticmethod
    def accumulate(target, metrics):
        for key, values in metrics.items():
            target.setdefault(key, OnlineStats()).add(values)

    def add_batch(self, start, batch):
        if start != self.written:
            raise ValueError("Batches must be written in manifest order, without skipping images.")
        zc, za = batch["zk_class"], batch["zk_agnostic"]
        n = len(zc)
        if zc.shape != za.shape or zc.shape[1] != 80 or start + n > len(self.samples):
            raise ValueError("Every image requires two matched 80-text banks.")
        for key, values in batch.items():
            self.norms.setdefault(key, OnlineStats()).add(np.linalg.norm(values, axis=-1))
        uc, ua = unit(zc), unit(za)
        for condition, values in [("class", uc), ("agnostic", ua)]:
            self.norms.setdefault("zk_unit_" + condition, OnlineStats()).add(np.linalg.norm(values, axis=-1))
        mc, ma = image_mean(zc), image_mean(za)
        metrics, means = paired_metrics(zc, za), paired_metrics(mc, ma)
        self.accumulate(self.paired, metrics)
        self.accumulate(self.mean_paired, means)
        dispersion_c, dispersion_a = prompt_dispersion(zc), prompt_dispersion(za)
        self.dispersion["class"].add(dispersion_c)
        self.dispersion["agnostic"].add(dispersion_a)
        self.vectors["visual"][start:start+n] = batch["visual"]
        self.vectors["e"][start:start+n] = batch["e"]
        self.vectors["zk_image_mean_class"][start:start+n] = mc
        self.vectors["zk_image_mean_agnostic"][start:start+n] = ma
        for key, array in self.all_z.items():
            array[start:start+n] = batch[key]
        e_unit = unit(batch["e"])
        current = self.samples[start:start+n]
        local_ids = [self.ids.index(s["class_id"]) for s in current]
        relevance_c = np.sum(batch["visual"][:, None] * self.pk["pk_class"][local_ids], axis=-1)
        relevance_a = batch["visual"] @ self.pk["pk_agnostic"].T
        cos_e_c, cos_e_a = np.sum(uc * e_unit[:, None], axis=-1), np.sum(ua * e_unit[:, None], axis=-1)
        norm_pre_c, norm_pre_a = np.linalg.norm(batch["zk_preln_class"], axis=-1), np.linalg.norm(batch["zk_preln_agnostic"], axis=-1)
        norm_up_c, norm_up_a = np.linalg.norm(batch["tda_update_class"], axis=-1), np.linalg.norm(batch["tda_update_agnostic"], axis=-1)
        individual, averaged = [], []
        for i, sample in enumerate(current):
            prefix = {"image_id": start+i, "class": sample["class"], "domain": sample["domain"]}
            self.paired_by_class[sample["class_id"]].add(means["cosine_shift"][i])
            self.paired_by_domain[sample["domain"]].add(means["cosine_shift"][i])
            for k in range(80):
                individual.append({**prefix, "template_id": k,
                    **{key: float(values[i, k]) for key, values in metrics.items()},
                    "cosine_z_E_class": float(cos_e_c[i, k]), "cosine_z_E_agnostic": float(cos_e_a[i, k]),
                    "cosine_pk_visual_class": float(relevance_c[i, k]), "cosine_pk_visual_agnostic": float(relevance_a[i, k]),
                    "norm_preLN_class": float(norm_pre_c[i, k]), "norm_preLN_agnostic": float(norm_pre_a[i, k]),
                    "norm_tda_update_class": float(norm_up_c[i, k]), "norm_tda_update_agnostic": float(norm_up_a[i, k])})
            averaged.append({**prefix, **{key: float(values[i]) for key, values in means.items()},
                             "prompt_dispersion_class": float(dispersion_c[i]),
                             "prompt_dispersion_agnostic": float(dispersion_a[i]),
                             "norm_E": float(np.linalg.norm(batch["e"][i]))})
        self.rows("zk_paired.csv", individual)
        self.rows("zk_image_means.csv", averaged)
        self.written += n

    def close(self):
        for handle in self.handles.values():
            handle.close()
        for array in list(self.vectors.values()) + list(self.all_z.values()):
            array.flush()

    def finish(self, make_tsne=True):
        self.close()
        if self.written != len(self.samples):
            raise ValueError("Incomplete full-dataset extraction: {}/{} images.".format(self.written, len(self.samples)))
        summary = summarize(self.pk, self.class_names, self.ids)
        summary["norms"].update({key: value.result() for key, value in self.norms.items()})
        summary["protocol"] = self.metadata
        summary["zk"] = {"paired_all_interventions": {k: v.result() for k, v in self.paired.items()},
                         "paired_image_means": {k: v.result() for k, v in self.mean_paired.items()},
                         "prompt_dispersion_class": self.dispersion["class"].result(),
                         "prompt_dispersion_agnostic": self.dispersion["agnostic"].result(),
                         "paired_image_shift_by_class": {self.class_names[c]: v.result() for c, v in self.paired_by_class.items()},
                         "paired_image_shift_by_domain": {d: v.result() for d, v in self.paired_by_domain.items()}}
        routes = {"V_baseline": "visual", "E_baseline": "e", "class_conditioned": "zk_image_mean_class", "class_agnostic": "zk_image_mean_agnostic"}
        by_domain, class_rows, pair_rows = {}, [], []
        for condition, key in routes.items():
            summary["zk"][condition] = exact_class_geometry(self.vectors[key], self.labels, self.class_names, self.domains)
            by_domain[condition] = {}
            for domain in sorted(set(self.domains.tolist())):
                mask = self.domains == domain
                by_domain[condition][domain] = exact_class_geometry(self.vectors[key][mask], self.labels[mask], self.class_names, self.domains[mask])
            for row in summary["zk"][condition]["per_class"]:
                class_rows.append({"condition": condition, "class": row["class"], "image_count": row["image_count"],
                                   "within_class_cosine": row["within_class_cosine"]["mean"] if row["within_class_cosine"] else None,
                                   "same_class_cross_domain_cosine": row["same_class_cross_domain_cosine"]["mean"] if row["same_class_cross_domain_cosine"] else None})
            pair_rows.extend({"condition": condition, **row} for row in summary["zk"][condition]["class_pairs"])
        write_csv(self.out / "class_metrics.csv", class_rows)
        write_csv(self.out / "class_pair_metrics.csv", pair_rows)
        write_csv(self.out / "feature_norm_statistics.csv", [{"feature": k, **v} for k, v in summary["norms"].items()])
        (self.out / "class_geometry_by_domain.json").write_text(json.dumps(by_domain, indent=2, allow_nan=False), encoding="utf-8")
        self.metadata["extraction_complete"] = True
        self.metadata["tsne_complete"] = False
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        # Reuse the small text-only report for all seven 80-text banks, without any image duplication.
        save_report(self.out / "pk", self.pk, self.class_names, self.ids,
                    {**self.metadata, "pk_only": True}, make_tsne=make_tsne)
        full_plots(self.out, summary, self.vectors, self.labels, self.domains, make_tsne)
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        lines = ["# Full-dataset paired pk/zk diagnostics", "",
                 "All {} local dataset images; {} classes; {} domains; 80 paired texts per image.".format(
                     len(self.samples), len(self.class_names), len(set(self.domains.tolist()))),
                 "No image/template sampling. Geometry is computed in the original feature dimension.", "",
                 "A pk point is one text. A zk t-SNE point is one image's normalized mean of its 80 unit zk vectors.",
                 "Image labels color the t-SNE map after fitting; they are not given to PCA/t-SNE.",
                 "B5B contexts use each image's true class for this descriptive feature analysis.", "",
                 "| Metric | V | E | B5B | B5A |", "|---|---:|---:|---:|---:|"]
        for metric in ["within_class_cosine", "between_class_cosine", "same_class_cross_domain_cosine"]:
            cells = [summary["zk"][c][metric] for c in routes]
            lines.append("| {} | {} |".format(metric, " | ".join("N/A" if v is None else "{:.6f}".format(v["mean"]) for v in cells)))
        lines += ["", "L2 norm measures one vector's length; L2 distance measures the difference of two vectors.",
                  "Unit pk norm is approximately 1 by construction; zk raw norm is constrained by LayerNorm.",
                  "A centroid is a class's image-feature mean. Cosine centroid distance is 1-cosine.",
                  "All-pair means are exact algebraic summaries, not sampled pairs; medians are not reported.",
                  "Micro metrics weight all image pairs; macro metrics weight classes equally.",
                  "The full dataset includes held-out-domain images for descriptive analysis only; this is not training or model selection.",
                  "The two text conditions use one fixed checkpoint; this is not a comparison of separately trained models."]
        (self.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary


def full_joint_tsne(groups, metadata):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits
    lengths = [len(g) for g in groups]
    combined = np.concatenate([unit(g).astype(np.float32) for g in groups])
    dimension = min(metadata.get("tsne_pca_dim", 50), len(combined)-1, combined.shape[1])
    with threadpool_limits(limits=4):
        if dimension:
            pca = PCA(n_components=dimension, random_state=metadata["seed"], svd_solver="randomized")
            combined = pca.fit_transform(combined)
            metadata["image_tsne_pca_explained_variance"] = float(pca.explained_variance_ratio_.sum())
        kwargs = dict(n_components=2, metric="euclidean", init="pca", random_state=metadata["seed"],
                      perplexity=min(float(metadata["tsne_perplexity"]), max(2, (len(combined)-1)/3)),
                      learning_rate=max(50, len(combined)/48), n_jobs=4)
        kwargs["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = metadata.get("tsne_iterations", 1000)
        coordinates = TSNE(**kwargs).fit_transform(combined)
    metadata["image_tsne_fit"] = {**kwargs, "joint_pca_dimensions": dimension,
                                  "point_count": len(combined), "image_sampling": False}
    return np.split(coordinates, np.cumsum(lengths)[:-1])


def full_plots(out, summary, vectors, labels, domains, make_tsne=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out = Path(out)
    routes = [("V_baseline", "Frozen CLIP V"), ("E_baseline", "E before TDA"),
              ("class_conditioned", "B5B class text"), ("class_agnostic", "B5A subject text")]
    names = summary["zk"]["E_baseline"]["classes"]

    def save(fig, filename):
        if summary["protocol"].get("synthetic_validation_only"):
            fig.text(.5, -.17, "SYNTHETIC DATA - rendering validation only", ha="center", color="firebrick")
        fig.savefig(out / (filename + ".png"), dpi=180, bbox_inches="tight")
        fig.savefig(out / (filename + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    for metric, filename, cmap, limit in [("centroid_cosine_distance", "image_class_centroid_distance", "Blues", (0, None)),
                                         ("same_class_domain_cosine_matrix", "same_class_domain_similarity", "viridis", (-1, 1))]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
        vmax = limit[1] if limit[1] is not None else max(.05, max(np.max(summary["zk"][c][metric]) for c, _ in routes))
        for ax, (condition, title) in zip(axes.ravel(), routes):
            matrix = np.asarray(summary["zk"][condition][metric])
            ticks = names if metric == "centroid_cosine_distance" else summary["zk"][condition]["domain_names"]
            im = ax.imshow(matrix, cmap=cmap, vmin=limit[0], vmax=vmax)
            ax.set_xticks(range(len(ticks)), ticks, rotation=35, ha="right")
            ax.set_yticks(range(len(ticks)), ticks)
            ax.set_title(title)
            for i in range(len(ticks)):
                for j in range(len(ticks)):
                    color = "white" if metric == "centroid_cosine_distance" and matrix[i,j] > vmax*.55 else "black"
                    ax.text(j, i, "{:.3f}".format(matrix[i,j]), ha="center", va="center", fontsize=8, color=color)
            fig.colorbar(im, ax=ax, fraction=.045)
        save(fig, filename)
    norm_groups = [("Raw text encoder", "pk_raw_class", "pk_raw_agnostic"),
                   ("pk after L2 normalization", "pk_class", "pk_agnostic"),
                   ("zk before LayerNorm", "zk_preln_class", "zk_preln_agnostic"),
                   ("zk after LayerNorm", "zk_class", "zk_agnostic")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (title, a, b) in zip(axes.ravel(), norm_groups):
        stats = [summary["norms"][key] for key in [a, b]]
        ax.bar([0,1], [s["mean"] for s in stats], yerr=[s["std"] for s in stats], capsize=4, color=["#88aaca", "#dfa795"])
        ax.set_xticks([0,1], ["B5B", "B5A"])
        ax.set_ylabel("L2 norm: mean and std")
        ax.set_title(title)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        if a == "pk_class":
            ax.set_ylim(.95, 1.05)
    save(fig, "feature_norms")
    if not make_tsne:
        return
    metadata = summary["protocol"]
    print("Joint t-SNE: all images, three conditions; PCA and t-SNE may take substantially longer than extraction.", flush=True)
    coordinates = full_joint_tsne([vectors["e"], vectors["zk_image_mean_class"], vectors["zk_image_mean_agnostic"]], metadata)
    np.savez_compressed(out / "image_tsne_coordinates.npz", E=coordinates[0], B5B=coordinates[1], B5A=coordinates[2], labels=labels, domains=domains)
    ids = summary["zk"]["E_baseline"]["class_ids"]
    class_colors = plt.get_cmap("tab10")
    domain_names = sorted(set(domains.tolist()))
    markers = ["o", "^", "s", "D"]
    point_size = max(3, min(12, 6000/len(labels)))
    for grouping, filename in [("class", "image_joint_tsne_by_class"), ("domain", "image_joint_tsne_by_domain")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True, constrained_layout=True)
        for ax, points, title in zip(axes, coordinates, ["E before TDA", "B5B class text", "B5A subject text"]):
            if grouping == "class":
                for local, c in enumerate(ids):
                    for d, domain in enumerate(domain_names):
                        mask = (labels == c) & (domains == domain)
                        ax.scatter(*points[mask].T, color=class_colors(local), marker=markers[d % 4], s=point_size, alpha=.35, rasterized=True)
            else:
                for d, domain in enumerate(domain_names):
                    mask = domains == domain
                    ax.scatter(*points[mask].T, color=class_colors(d), s=point_size, alpha=.35, rasterized=True)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        legend = ([Line2D([0],[0], marker="o", color=class_colors(i), linestyle="", label=name) for i,name in enumerate(names)]
                  if grouping == "class" else
                  [Line2D([0],[0], marker="o", color=class_colors(i), linestyle="", label=name) for i,name in enumerate(domain_names)])
        if grouping == "class":
            legend += [Line2D([0],[0], marker=markers[i % 4], color="gray", linestyle="", label=name)
                       for i,name in enumerate(domain_names)]
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5,-.08), ncol=min(7,len(legend)), fontsize=9)
        fig.suptitle("All {} images per condition; one joint fit; color by {}".format(len(labels), grouping))
        save(fig, filename)
    metadata["tsne_complete"] = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redraw full dataset figures from existing saved image vectors, without a GPU/checkpoint.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-tsne", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    summary = json.loads((out / "summary.json").read_text())
    with (out / "samples.csv").open(encoding="utf-8-sig") as handle:
        samples = list(csv.DictReader(handle))
    labels = np.asarray([int(s["class_id"]) for s in samples])
    domains = np.asarray([s["domain"] for s in samples])
    vectors = {key: np.load(out / "vectors" / (key+".npy"), mmap_mode="r")
               for key in ["visual", "e", "zk_image_mean_class", "zk_image_mean_agnostic"]}
    full_plots(out, summary, vectors, labels, domains, not args.no_tsne)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
