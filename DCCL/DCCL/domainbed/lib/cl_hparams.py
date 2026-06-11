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
    hparams["use_cirl"] = args.use_cirl
    hparams["lambda_cirl"] = args.lambda_cirl
    hparams["cirl_use_fourier_reliability"] = args.cirl_use_fourier_reliability
    hparams["cirl_reliability_temperature"] = args.cirl_reliability_temperature
    hparams["cirl_min_reliability"] = args.cirl_min_reliability
    hparams["cirl_fourier_alpha"] = args.cirl_fourier_alpha
    hparams["lambda_icr"] = args.lambda_icr
    hparams["use_rise"] = args.use_rise
    hparams["lambda_kd"] = 0.0 if args.disable_rise_kd else args.lambda_kd
    hparams["lambda_ad"] = 0.0 if args.disable_rise_ad else args.lambda_ad
    hparams["rise_prompt_mode"] = args.rise_prompt_mode
    hparams["rise_clip_model_name"] = args.rise_clip_model_name
    hparams["rise_clip_download_root"] = args.rise_clip_download_root
    hparams["rise_kd_temperature"] = args.rise_kd_temperature
    hparams["rise_ad_use_gt_label"] = args.rise_ad_use_gt_label
    hparams["rise_teacher_temperature"] = args.rise_teacher_temperature
    return hparams
