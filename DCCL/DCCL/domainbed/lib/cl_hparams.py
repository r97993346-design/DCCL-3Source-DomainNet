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
            # hparams["t"] = 0.1
            # hparams["t_pre"] = 0.3
            # hparams["l"] = 0.5
            # hparams["l_d"] = 0.05
            # hparams["l_layer"] = 0.5
            # hparams["n_layer"] = 1
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
            "cipt_tda_heads", "cipt_selector_mode",
            "cipt_selector_candidates", "cipt_selector_causal_penalty",
            "cipt_selector_js_weight", "cipt_selector_diversity_weight",
            "cipt_selector_warmup_steps", "cipt_contrastive_weight",
            "cipt_debug_shapes",
        ):
            hparams[name] = getattr(args, name)

    return hparams
