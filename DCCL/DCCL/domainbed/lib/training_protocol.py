"""Explicit configuration overrides and training-budget semantics."""

import math


CORE_HPARAMS = ("l", "l_d", "l_layer", "t", "t_pre", "n_layer")


def finalize_dccl_hparams(hparams, args, argv=()):
    # Unspecified argparse defaults must not override dataset/YAML values.
    explicit = {token.split("=", 1)[0] for token in argv}
    for key in CORE_HPARAMS:
        if "--" + key in explicit:
            hparams[key] = getattr(args, key)
    for key in ("l", "l_d", "l_layer"):
        value = float(hparams[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(key + " must be finite and nonnegative")
        hparams[key] = value
    for key in ("lr", "t", "t_pre"):
        value = float(hparams[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(key + " must be finite and positive")
        hparams[key] = value
    if hparams["n_layer"] not in (1, 2):
        raise ValueError("DCCL projection heads support n_layer=1 or 2")
    hparams["n_layer"] = int(hparams["n_layer"])
    return hparams


def resolve_update_budget(steps, checkpoint_freq, mode="exact"):
    if steps < 1 or checkpoint_freq < 1:
        raise ValueError("steps and checkpoint_freq must be positive")
    if mode == "exact":
        return steps
    if mode == "legacy":
        return (steps // checkpoint_freq) * checkpoint_freq + 1
    raise ValueError("step_count_mode must be exact or legacy")
