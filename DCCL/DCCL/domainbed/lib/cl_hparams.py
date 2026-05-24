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
    hparams["use_dfa"] = args.use_dfa
    hparams["l_cdc"] = args.l_cdc
    hparams["l_pma"] = args.l_pma
    hparams["l_gt"] = args.l_gt

    hparams["dfa_dim"] = args.dfa_dim
    hparams["dfa_use_dfd"] = args.dfa_use_dfd
    hparams["dfa_use_advm"] = args.dfa_use_advm
    hparams["dfa_use_dr"] = args.dfa_use_dr
    hparams["dfa_use_cr"] = args.dfa_use_cr
    hparams["dfa_mask_ratio"] = args.dfa_mask_ratio
    hparams["dfa_gumbel_tau"] = args.dfa_gumbel_tau
    hparams["lambda_dfa_cls"] = args.lambda_dfa_cls
    hparams["lambda_dfa_inv"] = args.lambda_dfa_inv
    hparams["lambda_dfa_cl"] = args.lambda_dfa_cl
    hparams["dfa_inv_rampup_epochs"] = args.dfa_inv_rampup_epochs
    hparams["dfa_inference_head"] = args.dfa_inference_head
    hparams["dfa_require_domain_balanced_batch"] = args.dfa_require_domain_balanced_batch
    hparams["dfa_log_fallback_stats"] = args.dfa_log_fallback_stats
    return hparams
