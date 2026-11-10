"""DICM-only configuration resolution; legacy DCCL resolution stays unchanged."""

import math


CORE_OVERRIDES = ("l", "l_layer", "t", "t_pre", "n_layer")


def finalize_dicm_hparams(hparams, args=None, argv=()):
    """Apply explicit argparse options after YAML/sconf, then enforce presets.

    A2 disables both contrasts. A3 disables PreCL. A1 is the no-GT identity
    baseline. A4 preserves configured DCCL task weights. No preset constructs GT.
    """
    explicit = {token.split("=", 1)[0] for token in argv}
    if args is not None:
        for name in CORE_OVERRIDES:
            if "--" + name in explicit:
                hparams[name] = getattr(args, name)
        if "--l_d" in explicit and float(args.l_d) != 0:
            raise ValueError("DICM removes GT: --l_d must be 0 (or omitted)")
    if float(hparams.get("l_d", 0)) != 0:
        raise ValueError("DICM removes GT: configuration l_d must be 0")
    mode = str(hparams.get("dicm_ablation", "A4")).upper()
    if mode not in {"A1", "A2", "A3", "A4"}:
        raise ValueError("dicm_ablation must be A1, A2, A3 or A4")
    hparams["dicm_ablation"] = mode
    if mode in {"A2", "A3"}:
        if args is not None and "--l_layer" in explicit and float(args.l_layer) != 0:
            raise ValueError(mode + " disables PreCL; do not pass nonzero --l_layer")
        hparams["l_layer"] = 0.0
    if mode == "A2":
        if args is not None and "--l" in explicit and float(args.l) != 0:
            raise ValueError("A2 disables SupCon; do not pass nonzero --l")
        hparams["l"] = 0.0
    hparams["l_d"] = 0.0
    for key in ("l", "l_layer", "dicm_isr_weight", "dicm_aug_weight", "dicm_dom_weight"):
        value = float(hparams.get(key, 0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(key + " must be finite and nonnegative")
        hparams[key] = value
    if mode != "A1" and float(hparams["dicm_aug_weight"]) + float(hparams["dicm_dom_weight"]) <= 0:
        raise ValueError("At least one ISR observation weight must be positive")
    for key in ("t", "t_pre", "lr"):
        if not math.isfinite(float(hparams[key])) or float(hparams[key]) <= 0:
            raise ValueError(key + " must be finite and positive")
    if int(hparams["n_layer"]) not in (1, 2):
        raise ValueError("DICM projection heads support n_layer=1 or 2")
    # These experimental legacy paths conflict with reliable source prototypes
    # or require different masks. Reject rather than silently bypass DICM.
    for key in ("TN", "re_w", "pos_mask", "aug", "mix"):
        if hparams.get(key, False):
            raise ValueError("DICM first version requires " + key + "=0/False")
    return hparams
