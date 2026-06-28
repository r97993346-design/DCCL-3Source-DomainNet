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
    hparams["dataset"] = args.dataset
    for key in [
        "use_causal_variant_factor", "causal_use_photometric", "causal_use_xdomainmix", "causal_use_diffusion",
        "causal_photo_ops", "causal_photo_strength_min", "causal_photo_strength_max",
        "causal_xdomainmix_alpha", "causal_xdomainmix_same_class_only", "causal_xdomainmix_require_diff_domain", "causal_xdomainmix_fallback_skip",
        "causal_diffusion_model_path", "causal_diffusion_local_only", "causal_diffusion_every_n_steps", "causal_diffusion_max_images_per_step",
        "causal_diffusion_steps", "causal_diffusion_cfg_text", "causal_diffusion_cfg_image", "causal_diffusion_seed", "causal_diffusion_device", "causal_style_per_image",
        "causal_prompt_mode", "causal_prompt_bank", "causal_use_pre_anchor_filter", "causal_pre_anchor_thresh",
        "causal_cls_filter_mode", "causal_cls_conf_thresh", "causal_filter_warmup_steps", "causal_cls_filter_use_ema",
        "causal_sensitivity_metric", "causal_sensitivity_temperature", "causal_top_m", "causal_sem_weight", "causal_kl_weight", "causal_kl_warmup_steps", "causal_cf_as_anchor",
        "causal_save_diffusion_images", "causal_save_diffusion_mode", "causal_save_diffusion_metadata", "causal_use_diffusion_cache", "causal_diffusion_cache_dir",
        "causal_filter_black_diffusion", "causal_black_mean_thresh", "causal_black_std_thresh",
    ]:
        hparams[key] = getattr(args, key)
    return hparams