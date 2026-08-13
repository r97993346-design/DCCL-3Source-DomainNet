from domainbed.datasets import datasets as datasets_registry


def _resolve_cipt_class_names(args):
    """Read class names from the same ImageFolder dataset used by DomainBed."""
    dataset_class = vars(datasets_registry)[args.dataset]
    dataset = dataset_class(args.data_dir)
    if hasattr(dataset, "datasets") and dataset.datasets:
        first = dataset.datasets[0]
        if hasattr(first, "classes"):
            return list(first.classes)
    return ["class {}".format(i) for i in range(dataset.num_classes)]


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

    if args.l_d is not None:
        hparams["l_d"] = args.l_d
    if args.l_layer is not None:
        hparams["l_layer"] = args.l_layer

    if args.algorithm == "CIPTDCCL":
        for name in (
            "cipt_enabled", "cipt_clip_backbone", "cipt_clip_path", "cipt_beta",
            "cipt_gamma", "cipt_k", "cipt_prompt_length", "cipt_prompt_init",
            "cipt_tda_heads", "cipt_contrastive_weight", "cipt_debug_shapes",
        ):
            hparams[name] = getattr(args, name)

    if args.algorithm == "CIPT":
        # TPAMI/public-code DG defaults. No DCCL loss/augmentation knobs are
        # consumed by the standalone CIPT implementation.
        hparams["cipt_clip_backbone"] = args.cipt_clip_backbone
        hparams["cipt_clip_path"] = args.cipt_clip_path
        hparams["cipt_beta"] = args.cipt_beta
        hparams["cipt_gamma"] = args.cipt_gamma
        hparams["cipt_k"] = args.cipt_k
        hparams["cipt_prompt_length"] = args.cipt_prompt_length
        hparams["cipt_prompt_init"] = args.cipt_prompt_init
        hparams["cipt_tda_heads"] = 8
        hparams["cipt_debug_shapes"] = args.cipt_debug_shapes
        hparams["cipt_lr"] = 2.5e-3
        hparams["cipt_weight_decay"] = 0.0
        dataset_class = vars(datasets_registry)[args.dataset]
        hparams["cipt_total_steps"] = int(args.steps or dataset_class.N_STEPS)
        hparams["cipt_class_names"] = _resolve_cipt_class_names(args)
        hparams["sample_d"] = False
        hparams["mix"] = 0
        hparams["aug"] = 0

    return hparams
