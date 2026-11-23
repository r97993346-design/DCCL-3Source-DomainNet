"""Class-conditional IND objectives for the direct E/S training path."""

import math
from numbers import Integral

import torch
import torch.nn.functional as F


IND_MODES = ("none", "original", "condcov", "coopood")


def resolve_ind_config(hparams):
    """Resolve the legacy switch only when no explicit mode was supplied."""
    mode = hparams.get("ind_mode")
    steps = hparams.get("ind_grad_steps")
    if steps is None:
        steps = ()
    objective = str(hparams.get("cipt_objective", "legacy")).lower()
    if not isinstance(steps, (list, tuple)) or any(
        isinstance(step, bool) or not isinstance(step, Integral) or step < 0 for step in steps
    ):
        raise ValueError("ind_grad_steps must be a list of non-negative update indices")
    if objective not in ("env_anchor", "env_kl"):
        if mode is not None or steps:
            raise ValueError("ind_mode and ind_grad_steps require env_anchor or env_kl")
        return None, 0.0, frozenset()
    if mode is None:
        mode = "original" if hparams.get("cipt_use_ind", True) else "none"
    if mode not in IND_MODES:
        raise ValueError(f"ind_mode must be one of {IND_MODES}")
    if mode == "none":
        weight = 0.0
    elif mode == "original":
        weight = float(hparams.get("cipt_gamma", 5.0))
    else:
        name = f"cipt_{mode}_weight"
        value = hparams.get(name)
        if value is None or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be explicitly set to a finite non-negative weight")
        weight = float(value)
    return mode, weight, frozenset(steps)


def _center_by_class(features, labels, membership, counts):
    means = (membership.T @ features) / counts.clamp_min(1).unsqueeze(1)
    return features - means[labels]


def conditional_covariance_loss(e, s, labels, num_classes):
    """Return the weighted squared Frobenius loss and number of valid classes.

    Labels are zero-based DomainBed training labels. Fixed-size masks keep all
    grouping on-device, including absent classes and singleton classes.
    """
    with torch.autocast(device_type=e.device.type, enabled=False):
        membership = (labels[:, None] == torch.arange(num_classes, device=labels.device)).float()
        counts = membership.sum(dim=0)
        valid = counts >= 2
        e_centered = _center_by_class(e.float(), labels, membership, counts)
        s_centered = _center_by_class(s.float(), labels, membership, counts)
        class_e = e_centered.T.unsqueeze(0) * membership.T.unsqueeze(1)
        cross = class_e @ s_centered
        cov = cross / (counts - 1).clamp_min(1)[:, None, None]
        weights = counts * valid / labels.numel()
        return (weights * cov.square().sum(dim=(1, 2))).sum(), valid.sum()


def coopood_loss(e, s, labels, num_classes):
    """CoOPood proxy: entrywise L1 of the pooled outer-product average."""
    with torch.autocast(device_type=e.device.type, enabled=False):
        membership = (labels[:, None] == torch.arange(num_classes, device=labels.device)).float()
        s_centered = _center_by_class(s.float(), labels, membership, membership.sum(dim=0))
        cross = (e.float().T @ s_centered) / labels.numel()
        return cross.abs().sum()


def feature_statistics(e, s, valid_classes=None):
    """Batch feature moments, with one device-to-host transfer."""
    metrics = {}
    with torch.no_grad():
        for name, feature in (("e", e), ("s", s)):
            feature = feature.float()
            metrics[f"{name}_feature_mean"] = feature.mean()
            metrics[f"{name}_feature_std"] = feature.std(unbiased=False)
            metrics[f"{name}_feature_batch_var"] = feature.var(dim=0, unbiased=False).mean()
        if valid_classes is not None:
            metrics["condcov_valid_classes"] = valid_classes.float()
        values = torch.stack(tuple(metrics.values())).tolist()
    return dict(zip(metrics, values))


def ind_gradient_diagnostics(e, s, loss_e, loss_s, loss_rd, loss_ind, weight):
    """Measure feature-space gradients without writing to parameter gradients."""
    gradients = {}
    for name, loss, features in (
        ("e_cls", loss_e, (e,)), ("s_uniform", loss_s, (s,)),
        ("rd", loss_rd, (e,)), ("selected_ind", loss_ind, (e, s)),
    ):
        if loss.requires_grad:
            gradients[name] = tuple(
                torch.zeros_like(feature) if gradient is None else gradient
                for feature, gradient in zip(features, torch.autograd.grad(
                    loss, features, retain_graph=True, allow_unused=True,
                ))
            )
        else:
            gradients[name] = tuple(torch.zeros_like(feature) for feature in features)

    cls_e, = gradients["e_cls"]
    rd_e, = gradients["rd"]
    uniform_s, = gradients["s_uniform"]
    ind_e, ind_s = gradients["selected_ind"]
    ind_e, ind_s = ind_e.float().flatten(), ind_s.float().flatten()
    metrics = {
        "ind_grad_e_cls_norm": cls_e.float().norm(),
        "ind_grad_e_rd_norm": rd_e.float().norm(),
        "ind_grad_s_uniform_norm": uniform_s.float().norm(),
        "ind_grad_e_selected_norm": ind_e.norm(),
        "ind_grad_s_selected_norm": ind_s.norm(),
        "ind_grad_e_selected_weighted_norm": weight * ind_e.norm(),
        "ind_grad_s_selected_weighted_norm": weight * ind_s.norm(),
        "ind_grad_e_cls_cosine": F.cosine_similarity(ind_e, cls_e.float().flatten(), dim=0),
        "ind_grad_e_rd_cosine": F.cosine_similarity(ind_e, rd_e.float().flatten(), dim=0),
        "ind_grad_s_uniform_cosine": F.cosine_similarity(ind_s, uniform_s.float().flatten(), dim=0),
    }
    return dict(zip(metrics, torch.stack(tuple(metrics.values())).tolist()))
