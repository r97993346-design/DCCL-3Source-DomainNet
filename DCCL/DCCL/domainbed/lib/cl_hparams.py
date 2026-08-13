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

    # Keep dataset metadata available to the DCCL->CIPT integration so class
    # names can be read from the same ImageFolder layout used by DomainBed.
    hparams["dataset"] = args.dataset
    hparams["data_dir"] = str(args.data_dir)

    # DCCL + official CIPT auxiliary objective.
    # CIPT internal defaults follow the official domain-generalization recipe:
    # beta=4, gamma=5, K=4. Only cipt_weight controls how strongly the entire
    # official CIPT objective regularizes DCCL, with a warm-up to avoid a large
    # auxiliary gradient before DCCL has formed a useful representation.
    hparams["cipt_weight"] = 0.05
    hparams["cipt_warmup_steps"] = 500
    hparams["cipt_beta"] = 4.0
    hparams["cipt_gamma"] = 5.0
    hparams["cipt_num_text_views"] = 4

    # Official CIPT prompt/TDA setup.
    hparams["cipt_n_ctx"] = 16
    hparams["cipt_ctx_init"] = "a photo of a"
    hparams["cipt_sample_templates"] = True
    hparams["cipt_tda_heads"] = 8
    hparams["cipt_tda_dropout"] = 0.0

    # Official CIPT optimizer recipe for prompt/adapters/TDA. DCCL keeps its
    # original optimizer; the fusion bridge is trained with the CIPT optimizer.
    hparams["cipt_lr"] = 2.5e-3
    hparams["cipt_weight_decay"] = 0.0

    # OpenAI CLIP text-side model used by official CIPT. cipt_clip_path can
    # point to a local checkpoint to avoid downloading by model name.
    hparams["cipt_clip_model"] = "ViT-B/16"
    hparams["cipt_clip_path"] = ""
    hparams["cipt_clip_download_root"] = ""

    # Optional comma-separated override. Normally class names are discovered
    # from data_dir/<dataset>/<environment>/<class>/ in ImageFolder order.
    hparams["cipt_classnames"] = ""
    return hparams
