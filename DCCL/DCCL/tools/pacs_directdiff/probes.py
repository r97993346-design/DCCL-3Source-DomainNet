"""Paired, source-only linear probes with one shared z scaler."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


PRIMARY_C = 10.0
SENSITIVITY_C = 100.0


def z_standardizer(z, train):
    source = z[train].astype(np.float64, copy=False)
    center = source.mean(0)
    spread = source.std(0)
    return center, np.maximum(spread, max(spread.max() * 1e-6, 1e-8))


def fit_probe(array, y, train, validation, center, spread, regularization):
    source = np.ascontiguousarray((array[train] - center) / spread)
    target = np.ascontiguousarray((array[validation] - center) / spread)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        for budget in (4000, 16000):
            classifier = LogisticRegression(solver="lbfgs", C=regularization, max_iter=budget, tol=1e-7)
            try:
                classifier.fit(source, y[train])
                break
            except ConvergenceWarning:
                if budget == 16000:
                    raise
    predicted = classifier.predict(target)
    return classifier, dict(
        accuracy=float(np.mean(predicted == y[validation])),
        balanced_accuracy=float(balanced_accuracy_score(y[validation], predicted)),
        iterations=int(classifier.n_iter_.max()), max_iterations=budget,
        train_samples=len(source), validation_samples=len(target), C=regularization,
    )


def probe_views(views, z, payload, affine_maps=None,
                sensitivity=("z", "gate", "uniform_equal_train_energy", "uniform_equal_sample_energy")):
    """Fit source probes under shared scaling and measure exact affine readout transport."""
    center, spread = z_standardizer(z, payload["train"])
    scores, transports = {}, {}
    tasks = (
        ("class", payload["y"], payload["train"], ~payload["train"]),
        ("domain", payload["domain"], payload["domain_train"], payload["domain_validation"]),
    )
    for name, array in views.items():
        scores[name] = {}
        for task, labels, train, validation in tasks:
            reg_values = (PRIMARY_C, SENSITIVITY_C) if name in sensitivity else (PRIMARY_C,)
            fits = {}
            for reg in reg_values:
                model, result = fit_probe(array, labels, train, validation, center, spread, reg)
                fits[str(reg)] = result
                if affine_maps and name == "z" and reg == PRIMARY_C:
                    w = (model.coef_ / spread).T
                    bias = model.intercept_ - center @ w
                    scores_z = z[validation] @ w + bias
                    transports[task] = {}
                    for candidate, (basis, attenuation) in affine_maps.items():
                        ratio = attenuation / (1 - attenuation)
                        weights = w + basis @ ((basis.T @ w) * ratio[:, None])
                        from_view = views[candidate][validation] @ weights + bias
                        transports[task][candidate] = dict(
                            max_abs_logit_error=float(np.abs(from_view - scores_z).max()),
                            prediction_agreement=float((from_view.argmax(1) == scores_z.argmax(1)).mean()),
                        )
            scores[name][task] = fits
    return scores, transports
