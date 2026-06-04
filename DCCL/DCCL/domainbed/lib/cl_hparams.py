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
    hparams["use_rise"] = args.use_rise
    hparams["use_rise_kd"] = args.use_rise_kd
    hparams["use_rise_proto"] = args.use_rise_proto
    hparams["rise_clip_model_name"] = args.rise_clip_model_name
    hparams["rise_clip_download_root"] = args.rise_clip_download_root
    hparams["rise_kd_weight"] = args.rise_kd_weight
    hparams["rise_proto_weight"] = args.rise_proto_weight
    hparams["rise_kd_temperature"] = args.rise_kd_temperature
    hparams["rise_prompt_mode"] = args.rise_prompt_mode
    hparams["rise_freeze_clip"] = args.rise_freeze_clip
    hparams["rise_projection_dim"] = args.rise_projection_dim
    return hparams
