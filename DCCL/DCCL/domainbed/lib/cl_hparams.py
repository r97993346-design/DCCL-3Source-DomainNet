import math

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
    if args.dataset == "PACS":
        hparams["t"] = 0.1
        hparams["t_pre"] = 0.2
        hparams["l"] = 1
        hparams["l_d"] = 0.01
        hparams["l_layer"] = 1
        hparams["n_layer"] = 1
    elif args.dataset == "TerraIncognita":
        hparams["t"] = 0.1
        hparams["t_pre"] = 0.1
        hparams["l"] = 1
        hparams["l_d"] = 0.05
        hparams["l_layer"] = 0.1
        hparams["n_layer"] = 2
    elif args.dataset == "VLCS":
        hparams["t"] = 0.1
        hparams["t_pre"] = 0.2
        hparams["l"] = 1
        hparams["l_d"] = 0.05
        hparams["l_layer"] = 1
        hparams["n_layer"] = 1
    elif args.dataset == "OfficeHome":
        if args.model == "clip_vit-b16":
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
    elif args.dataset == "DomainNet":
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
            "cipt_enabled",
            "cipt_clip_backbone",
            "cipt_clip_path",
            "cipt_beta",
            "cipt_gamma",
            "cipt_k",
            "cipt_prompt_length",
            "cipt_prompt_init",
            "cipt_tda_heads",
            "cipt_contrastive_weight",
            "cipt_debug_shapes",
        ):
            hparams[name] = getattr(args, name)

    if args.algorithm == "CIPT":
        # TPAMI/public-code CIPT DG settings.
        class_names = _resolve_cipt_class_names(args)
        dataset_class = vars(datasets_registry)[args.dataset]

        shots = 16
        epochs = 30
        global_batch_size = 64
        num_sources = (
            len(args.source_envs)
            if args.dataset == "DomainNet" and args.source_envs is not None
            else max(1, len(dataset_class.ENVIRONMENTS) - 1)
        )

        # DomainBed produces one equal-sized minibatch per source domain.
        # Choose the smallest per-domain batch whose merged batch reaches 64;
        # the thin CIPT adapter trims the merged batch to exactly 64.
        per_domain_batch = int(math.ceil(global_batch_size / float(num_sources)))
        examples_per_source_epoch = shots * len(class_names)
        steps_per_epoch = max(
            1,
            int(round(examples_per_source_epoch / float(per_domain_batch))),
        )
        paper_total_steps = epochs * steps_per_epoch

        hparams["cipt_official"] = True
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
        hparams["cipt_shots"] = shots
        hparams["cipt_epochs"] = epochs
        hparams["cipt_global_batch_size"] = global_batch_size
        hparams["cipt_steps_per_epoch"] = steps_per_epoch
        hparams["cipt_total_steps"] = paper_total_steps
        hparams["cipt_class_names"] = class_names

        # Pure CIPT does not use any DCCL sampling/mixing branch.
        hparams["sample_d"] = False
        hparams["mix"] = 0
        hparams["aug"] = 0
        hparams["batch_size"] = per_domain_batch

        # IMPORTANT: generic DomainBed --steps / --checkpoint_freq belong to the
        # old step-based trainer. For the official CIPT baseline, always enforce
        # exactly 30 epochs and evaluate once per CIPT epoch. This prevents an
        # old DCCL command such as --steps 5000 --checkpoint_freq 100 from
        # silently turning 30 epochs into roughly 1000 epochs.
        args.steps = paper_total_steps
        args.checkpoint_freq = steps_per_epoch

    return hparams
