"""DICM preset resolution after dataset/YAML configuration."""

import math


CORE_OVERRIDES = ("l", "l_layer", "t", "t_pre", "n_layer")


def finalize_dicm_hparams(hparams, args=None, argv=()):
    """Apply explicit argparse options after YAML/sconf, then enforce presets.

    A2 disables both contrasts. A3 disables PreCL. A1 is the no-GT identity
    baseline. A4 preserves configured DCCL task weights without GT. Only A5
    reintroduces GT as a complementary full-model control.
    """
    explicit = {token.split("=", 1)[0] for token in argv}
    if args is not None:
        for name in CORE_OVERRIDES:
            if "--" + name in explicit:
                hparams[name] = getattr(args, name)
    mode = str(hparams.get("dicm_ablation", "A4")).upper()
    if mode not in {"A1", "A2", "A3", "A4", "A5"}:
        raise ValueError("dicm_ablation must be A1, A2, A3, A4 or A5")
    hparams["dicm_ablation"] = mode
    if mode == "A5":
        gt_weight = float(hparams.get("dicm_gt_weight", 0.01))
        if args is not None and "--l_d" in explicit:
            gt_weight = float(args.l_d)
        if not math.isfinite(gt_weight) or gt_weight <= 0:
            raise ValueError("A5 requires a finite positive GT weight; use A4 to disable GT")
        previous_ld = float(hparams.get("l_d", 0))
        if ("--l_d" not in explicit and "--dicm_gt_weight" not in explicit
                and previous_ld not in (0.0, gt_weight)):
            raise ValueError("A5: set GT with dicm_gt_weight; l_d must be 0 or the same value")
        # l_d is the effective coefficient in logs/checkpoints, not argparse's default.
        hparams["dicm_gt_weight"] = hparams["l_d"] = gt_weight
    else:
        if args is not None and "--l_d" in explicit and float(args.l_d) != 0:
            raise ValueError("A1-A4 remove GT: --l_d must be 0 (or omitted)")
        if float(hparams.get("l_d", 0)) != 0:
            raise ValueError("A1-A4 remove GT: configuration l_d must be 0")
        hparams["l_d"] = 0.0
    if mode in {"A2", "A3"}:
        if args is not None and "--l_layer" in explicit and float(args.l_layer) != 0:
            raise ValueError(mode + " disables PreCL; do not pass nonzero --l_layer")
        hparams["l_layer"] = 0.0
    if mode == "A2":
        if args is not None and "--l" in explicit and float(args.l) != 0:
            raise ValueError("A2 disables SupCon; do not pass nonzero --l")
        hparams["l"] = 0.0
    for key in ("l", "l_layer", "dicm_isr_weight", "dicm_aug_weight", "dicm_dom_weight"):
        value = float(hparams.get(key, 0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(key + " must be finite and nonnegative")
        hparams[key] = value
    if mode == "A5" and (hparams["l"] <= 0 or hparams["l_layer"] <= 0):
        raise ValueError("A5 is the full model plus GT; SupCon and PreCL must both be enabled")
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
