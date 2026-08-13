import sys


def _cli_option_was_provided(name):
    """Return True for both `--name value` and `--name=value` forms."""
    flag = "--" + name
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def setup_alg_hparams(hparams, args):
    if args.dataset=="PACS":
        hparams["t"] = 0.1
        hparams["t_pre"] = 0.2
        hparams["l"] = 1
        hparams["l_d"] = 0.01
        hparams["l_layer"] = 1
        hparams["n_layer"] = 1
    elif args.dataset=="TerraIncognita":
        hparams["t"] = 0.1
        hparams["t_pre"] = 0.1
        hparams["l"] = 1
        hparams["l_d"] = 0.05
        hparams["l_layer"] = 0.1
        hparams["n_layer"] = 2
    elif args.dataset=="VLCS":
        hparams["t"] = 0.1
        hparams["t_pre"] = 0.2
        hparams["l"] = 1
        hparams["l_d"] = 0.05
        hparams["l_layer"] = 1
        hparams["n_layer"] = 1
    elif args.dataset=="OfficeHome":
        if args.model=="clip_vit-b16":
            hparams["t"] = 0.2
            hparams["t_pre"] = 0.2
            hparams["l"] = 1
            hparams["l_d"] = 0.1
            hparams["l_layer"] = 1
            hparams["n_layer"] = 1
        else:
            hparams["t"] = 0.1
            hparams["t_pre"] = 0.3
            hparams["l"] = 1
            hparams["l_d"] = 0.05
            hparams["l_layer"] = 5
            hparams["n_layer"] = 1
    elif args.dataset=="DomainNet":
        hparams["t"] = 0.1
        hparams["t_pre"] = 0.1
        hparams["l"] = 1
        hparams["l_d"] = 0.05
        hparams["l_layer"] = 0.1
        hparams["n_layer"] = 2

    hparams["sup"] = args.sup
    hparams["two_ce"] = args.two_ce
    hparams["sample_d"] = args.sample_d
    hparams["re_w"] = args.re_w
    hparams["pos_mask"] = args.pos_mask
    hparams["mix"] = args.mix
    hparams["aug"] = args.aug
    hparams["model"] = args.model
    hparams["label_ratio"] = args.label_ratio
    hparams["TN"] = args.TN
    hparams["lamda"] = args.lamda
    hparams["start_epoch"] = args.start_epoch
    hparams["log"] = args.log

    # Explicit CLI overrides for DCCL ablation experiments.
    # Keep dataset-specific defaults when the CLI option is not provided.
    if args.l_d is not None:
        hparams["l_d"] = args.l_d
    if args.l_layer is not None:
        hparams["l_layer"] = args.l_layer

    if args.algorithm == "CIPTDCCL":
        for name in (
            "cipt_enabled", "cipt_clip_backbone", "cipt_clip_path", "cipt_beta",
            "cipt_gamma", "cipt_k", "cipt_prompt_length", "cipt_prompt_init",
            "cipt_contrastive_weight", "cipt_debug_shapes",
        ):
            hparams[name] = getattr(args, name)

        # Official public CIPT defaults to 8 attention heads. train_all.py kept
        # the historical parser default=1, so distinguish an omitted flag from
        # an explicitly requested `--cipt_tda_heads 1` ablation.
        if _cli_option_was_provided("cipt_tda_heads"):
            hparams["cipt_tda_heads"] = int(args.cipt_tda_heads)
        else:
            hparams["cipt_tda_heads"] = 8

        # Official CIPT prompt/adapters/TDA optimizer defaults.
        hparams["cipt_lr"] = 2.5e-3
        hparams["cipt_weight_decay"] = 0.0

        # DomainBed is step-based. Use the requested run horizon (or the dataset
        # default) for a step-wise cosine decay of only the CIPT optimizer group.
        if args.steps is not None:
            hparams["cipt_schedule_steps"] = max(1, int(args.steps))
        elif args.dataset == "DomainNet":
            hparams["cipt_schedule_steps"] = 15001
        else:
            hparams["cipt_schedule_steps"] = 5001

    return hparams
