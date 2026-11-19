"""Source-only linear probes, random-basis controls, and class-conditional geometry."""
from itertools import combinations
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from tools.pacs_causal.features import STYLES, project


RANDOM_BASES = 12
PROBE = dict(solver="lbfgs", C=1.0, max_iter=1000, tol=1e-5)


def fit_probe(data, labels, fit, score):
    train, validation = data[fit], data[score]
    # One nearest-common scalar per representation keeps directions and rank intact.
    center = train.mean(0, dtype=np.float64).astype(data.dtype)
    train = train - center
    scale = np.sqrt(np.mean(np.einsum("nd,nd->n", train, train)))
    if scale <= 1e-12:
        raise ValueError("Degenerate representation in a probe")
    model = LogisticRegression(**PROBE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train / scale, labels[fit])
    predicted = model.predict((validation - center) / scale)
    return dict(acc=float((predicted == labels[score]).mean()),
                macro=float(balanced_accuracy_score(labels[score], predicted)),
                train_n=int(len(train)), validation_n=int(len(validation)),
                converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught))


def probe_pair(data, payload):
    train = payload["train"]
    category = fit_probe(data, payload["y"], train, ~train)
    domain = fit_probe(data, payload["env"], payload["domain_train"], payload["domain_validation"])
    return dict(category=category, domain=domain)


def energy(x):
    return float(np.einsum("nd,nd->", x, x, dtype=np.float64))


def energy_metrics(reps, z_ref, q):
    z, e = reps["z"], reps["E_beta"]
    s, r = reps["S_removed"], z - z_ref
    hard, orth = reps["E_full_res"], reps["E_orth"]
    dot = lambda a, b: np.einsum("nd,nd->n", a, b)
    alignment = np.abs(dot(e, s)) / np.maximum(np.sqrt(dot(e, e) * dot(s, s)), 1e-12)
    same = (np.linalg.norm((hard - orth) - project(z_ref, q)) / max(np.linalg.norm(z), 1e-12))
    return dict(retained_energy_over_z=energy(e) / energy(z),
                removed_energy_over_z=energy(s) / energy(z),
                hard_res_energy_over_z=energy(hard) / energy(z),
                orth_energy_over_z=energy(orth) / energy(z),
                residual_energy_captured=energy(reps["S_raw"]) / energy(r),
                mean_abs_cos_E_beta_removed=float(alignment.mean()),
                e_orth_projected_energy_fraction=energy(orth @ q) / energy(z),
                e_full_res_projected_energy_fraction=energy(hard @ q) / energy(z),
                relative_error_E_full_minus_E_orth_vs_projected_reference=float(same))


def norm_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def scatter_stats(x):
    center = x.mean(0)
    scatter = np.einsum("nd,nd->", x - center, x - center) / len(x)
    return center, float(scatter)


def geometry(data, y, d, axes):
    """Each class and pair of source domains receives equal weight."""
    all_classes, all_domains = np.unique(y), np.unique(d)
    raw, normalized, frobenius = [], [], []
    class_centroids, within_class = [], []
    low = axes.transform(data)
    normalized_data = norm_rows(data)
    for category in all_classes:
        by_domain = [np.flatnonzero((y == category) & (d == domain)) for domain in all_domains]
        if any(len(cell) < 2 for cell in by_domain):
            raise ValueError("Covariance comparison requires all source classes in all domains")
        raw_stats = [scatter_stats(data[cell]) for cell in by_domain]
        unit_stats = [scatter_stats(normalized_data[cell]) for cell in by_domain]
        covariances = [np.cov(low[cell], rowvar=False) for cell in by_domain]
        class_centroids.append(np.mean([x[0] for x in unit_stats], axis=0))
        class_scatter = []
        for center, scatter in unit_stats:
            difference = center - class_centroids[-1]
            class_scatter.append(scatter + float(difference @ difference))
        within_class.append(np.mean(class_scatter))
        for a, b in combinations(range(len(by_domain)), 2):
            def squared_pair_mean(stats):
                difference = stats[a][0] - stats[b][0]
                return float(difference @ difference + stats[a][1] + stats[b][1])
            raw.append(squared_pair_mean(raw_stats))
            normalized.append(squared_pair_mean(unit_stats))
            denom = np.linalg.norm(covariances[a]) + np.linalg.norm(covariances[b])
            frobenius.append(np.linalg.norm(covariances[a] - covariances[b]) / max(denom, 1e-12))
    between = float(np.mean([np.sum((a-b)**2) for a, b in combinations(class_centroids, 2)]))
    within = float(np.mean(within_class))
    return dict(same_class_cross_domain_ms_l2=float(np.mean(raw)),
                same_class_cross_domain_ms_unit_l2=float(np.mean(normalized)),
                same_class_covariance_relative_frobenius_pca48=float(np.mean(frobenius)),
                between_class_centroid_ms_unit_l2=between, within_class_unit_scatter=within,
                between_over_within_class=float(between / max(within, 1e-12)))


def coverage(delta, q):
    denom = np.einsum("nd,nd->n", delta, delta)
    valid = denom > 1e-8
    if not valid.any():
        raise ValueError("No valid held-out intervention responses")
    coeff = delta[valid] @ q
    ratios = np.einsum("nr,nr->n", coeff, coeff) / denom[valid]
    return dict(mean=float(ratios.mean()), median=float(np.median(ratios)),
                p10=float(np.quantile(ratios, .1)), p90=float(np.quantile(ratios, .9)),
                valid=int(valid.sum()))


def analyze(payload):
    reps, q, train = payload["representations"], payload["q"], payload["train"]
    probes = {key: probe_pair(value, payload) for key, value in reps.items()}
    view = ~train
    reference_axes = PCA(n_components=48, svd_solver="randomized", random_state=0).fit(reps["z"][train])
    distances = {
        key: geometry(reps[key][view], payload["y"][view], payload["env"][view], reference_axes)
        for key in ("z", "E_beta", "E_full_res", "E_orth")
    }
    observations = []
    changes = []
    residual = reps["z"] - payload["z_ref"]
    for index in range(-1, RANDOM_BASES):
        seed = 202611190 + 100 * payload["target"] + index
        if index < 0:
            label, seed, basis = "learned", None, q
        else:
            label = f"random_{index:02d}"
            rng = np.random.default_rng(seed)
            matrix = rng.standard_normal(q.shape, dtype=np.float32)
            basis = np.linalg.qr(matrix, mode="reduced")[0]
        sensitive = project(residual, basis)
        stable = reps["z"] - payload["beta"] * sensitive
        if index < 0:
            stable_probes, sensitive_probes = probes["E_beta"], probes["S_removed"]
        else:
            stable_probes = probe_pair(stable, payload)
            sensitive_probes = probe_pair(reps["z"] - stable, payload)
        observations.append(dict(q=label, seed=seed,
                                 stable_category=stable_probes["category"]["acc"],
                                 stable_domain=stable_probes["domain"]["acc"],
                                 removed_category=sensitive_probes["category"]["acc"],
                                 removed_domain=sensitive_probes["domain"]["acc"],
                                 stable_category_converged=stable_probes["category"]["converged"],
                                 stable_domain_converged=stable_probes["domain"]["converged"],
                                 removed_category_converged=sensitive_probes["category"]["converged"],
                                 removed_domain_converged=sensitive_probes["domain"]["converged"],
                                 removed_energy_over_z=energy(reps["z"][view]-stable[view])
                                 / energy(reps["z"][view])))
        for style in STYLES:
            for name, response in (("delta_r", payload["dr"][style]), ("delta_z", payload["dz"][style])):
                changes.append(dict(q=label, seed=seed, style=style, response=name,
                                    **coverage(response, basis)))

    val_features = {key: array[view] for key, array in reps.items()}
    energies = energy_metrics(val_features, payload["z_ref"][view], q)
    return dict(probes=probes, geometry=distances, q_controls=observations, perturbations=changes,
                energies=energies, source_validation_alpha=payload["alpha"],
                covariance_explained_variance=float(reference_axes.explained_variance_ratio_.sum()),
                probe_configuration=PROBE)
