"""Converged source-only linear ridge readouts with one shared z scaler."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import balanced_accuracy_score


PRIMARY_RIDGE = 1e-3
SENSITIVITY_RIDGE = 1e-4


def z_standardizer(z, train):
    source = z[train].astype(np.float64, copy=False)
    center = source.mean(0)
    spread = source.std(0)
    return center, np.maximum(spread, max(spread.max() * 1e-6, 1e-8))


def fit_probe(array, y, train, validation, center, spread, ridge):
    """Solve a one-hot ridge regression with centered features and a free intercept."""
    source = np.ascontiguousarray((array[train] - center) / spread)
    classes, tags = np.unique(y[train], return_inverse=True)
    targets = np.eye(len(classes))[tags]
    y_mean = targets.mean(0)
    x_mean = source.mean(0)
    source -= x_mean
    targets -= y_mean
    n, width = source.shape
    strength = n * ridge
    if n < width:
        gram = source @ source.T
        gram.flat[::n + 1] += strength
        weight = source.T @ cho_solve(cho_factor(gram, lower=True), targets)
    else:
        gram = source.T @ source
        gram.flat[::width + 1] += strength
        weight = cho_solve(cho_factor(gram, lower=True), source.T @ targets)
    # Verify the first-order optimum on the actual training data for every fit.
    gradient = source.T @ (source @ weight - targets) / n + ridge * weight
    reference = np.linalg.norm(source.T @ targets / n)
    residual = float(np.linalg.norm(gradient) / max(reference, 1e-12))
    if not np.isfinite(residual) or residual > 1e-7:
        raise FloatingPointError(f"Linear ridge solve failed the normal-equation check: {residual}")
    raw = weight / spread[:, None]
    bias = y_mean - (center / spread + x_mean) @ weight
    predicted = classes[(array[validation] @ raw + bias).argmax(1)]
    return (raw, bias), dict(
        accuracy=float(np.mean(predicted == y[validation])),
        balanced_accuracy=float(balanced_accuracy_score(y[validation], predicted)),
        train_samples=n, validation_samples=len(predicted), ridge=ridge,
        normal_equation_relative_residual=residual,
    )


def probe_views(views, z, payload, affine_maps=None,
                sensitivity=("z", "gate", "uniform_equal_train_energy", "uniform_equal_sample_energy")):
    """Fit source probes under shared scaling and test affine readout transport."""
    center, spread = z_standardizer(z, payload["train"])
    scores, transports = {}, {}
    tasks = (
        ("class", payload["y"], payload["train"], ~payload["train"]),
        ("domain", payload["domain"], payload["domain_train"], payload["domain_validation"]),
    )
    for name, array in views.items():
        scores[name] = {}
        for task, labels, train, validation in tasks:
            regularizations = (PRIMARY_RIDGE, SENSITIVITY_RIDGE) if name in sensitivity else (PRIMARY_RIDGE,)
            fits = {}
            for reg in regularizations:
                (weight, bias), result = fit_probe(array, labels, train, validation, center, spread, reg)
                fits[str(reg)] = result
                if affine_maps and name == "z" and reg == PRIMARY_RIDGE:
                    scores_z = z[validation] @ weight + bias
                    transports[task] = {}
                    for candidate, (basis, attenuation) in affine_maps.items():
                        ratio = attenuation / (1 - attenuation)
                        moved = weight + basis @ ((basis.T @ weight) * ratio[:, None])
                        from_view = views[candidate][validation] @ moved + bias
                        transports[task][candidate] = dict(
                            max_abs_logit_error=float(np.abs(from_view - scores_z).max()),
                            prediction_agreement=float((from_view.argmax(1) == scores_z.argmax(1)).mean()),
                        )
            scores[name][task] = fits
    return scores, transports
