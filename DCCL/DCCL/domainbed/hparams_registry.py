# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np


def _hparams(algorithm, dataset, random_state):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ["Debug28", "RotatedMNIST", "ColoredMNIST"]

    hparams = {}

    hparams["data_augmentation"] = (True, True)
    hparams["val_augment"] = (False, False)  # augmentation for in-domain validation set
    hparams["resnet18"] = (False, False)
    hparams["resnet_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5]))
    hparams["class_balanced"] = (False, False)
    hparams["optimizer"] = ("adam", "adam")

    if algorithm == "CIPTDCCL":
        hparams["cipt_enabled"] = (True, True)
        hparams["cipt_clip_backbone"] = ("ViT-B/16", "ViT-B/16")
        hparams["cipt_clip_path"] = ("", "")
        hparams["cipt_beta"] = (4.0, 4.0)
        hparams["cipt_gamma"] = (5.0, 5.0)
        hparams["cipt_k"] = (4, 4)
        hparams["cipt_prompt_length"] = (16, 16)
        hparams["cipt_prompt_init"] = ("a photo of a", "a photo of a")
        # TDA modes for the paired class-agnostic validation protocol:
        # b5b    = B0     : class-conditioned + diverse B5b contexts
        # b5a    = S0     : paired class-agnostic contexts
        # bconst = Bconst : 80x "a photo of a {class}."
        # sconst = Sconst : 80x "a photo of a <neutral subject>."
        # b5c             : legacy expanded class-agnostic prompt bank
        hparams["cipt_template_mode"] = ("b5a", "b5a")
        # Lexical robustness factor for S0/Sconst. Valid implementation choices:
        # subject / thing / object / entity.
        hparams["cipt_neutral_subject"] = ("subject", "subject")
        hparams["cipt_tda_heads"] = (1, 1)
        # Keep paired B0/S0 validation unchanged unless explicitly enabled.
        # Adaptive selection uses the full B5a/B5c bank in train and eval.
        hparams["cipt_selector_mode"] = ("random", "random")
        hparams["cipt_selector_candidates"] = (8, 8)
        hparams["cipt_contrastive_weight"] = (1.0, 1.0)
        # Independent causal-contrastive search knobs.  They are intentionally
        # not tied to the standard DCCL/SupCon hyperparameters.
        hparams["cipt_contrastive_temperature"] = (0.1, 0.1)
        hparams["cipt_contrastive_type"] = ("neighbor_retention", "neighbor_retention")
        hparams["cipt_neighbor_k"] = (5, 5)
        hparams["cipt_neighbor_alpha"] = (0.5, 0.5)
        hparams["cipt_neighbor_diagnostics"] = (False, False)
        hparams["cipt_use_aug_view"] = (False, False)
        hparams["cipt_debug_shapes"] = (False, False)

    hparams["freeze_bn"] = (True, True)
    hparams["pretrained"] = (True, True)  # only for ResNet

    if dataset not in SMALL_IMAGES:
        hparams["lr"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        if dataset == "DomainNet":
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5)))
        else:
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5.5)))
        if algorithm == "ARM":
            hparams["batch_size"] = (8, 8)
    else:
        hparams["lr"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
        hparams["batch_size"] = (64, int(2 ** random_state.uniform(3, 9)))

    if dataset in SMALL_IMAGES:
        hparams["weight_decay"] = (0.0, 0.0)
    else:
        hparams["weight_decay"] = (0.0, 10 ** random_state.uniform(-6, -2))

    if algorithm in ["DANN", "CDANN"]:
        if dataset not in SMALL_IMAGES:
            hparams["lr_g"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
            hparams["lr_d"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        else:
            hparams["lr_g"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
            hparams["lr_d"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))

        if dataset in SMALL_IMAGES:
            hparams["weight_decay_g"] = (0.0, 0.0)
        else:
            hparams["weight_decay_g"] = (0.0, 10 ** random_state.uniform(-6, -2))

        hparams["lambda"] = (1.0, 10 ** random_state.uniform(-2, 2))
        hparams["weight_decay_d"] = (0.0, 10 ** random_state.uniform(-6, -2))
        hparams["d_steps_per_g_step"] = (
            1,
            int(2 ** random_state.uniform(0, 3)),
        )
        hparams["grad_penalty"] = (0.0, 10 ** random_state.uniform(-2, 1))
        hparams["beta1"] = (0.5, random_state.choice([0.0, 0.5]))
    elif algorithm == "RSC":
        hparams["rsc_f_drop_factor"] = (1 / 3, random_state.uniform(0, 0.5))
        hparams["rsc_b_drop_factor"] = (1 / 3, random_state.uniform(0, 0.5))
    elif algorithm == "SagNet":
        hparams["sag_w_adv"] = (0.1, 10 ** random_state.uniform(-2, 1))
    elif algorithm == "IRM":
        hparams["irm_lambda"] = (1e2, 10 ** random_state.uniform(-1, 5))
        hparams["irm_penalty_anneal_iters"] = (
            500,
            int(10 ** random_state.uniform(0, 4)),
        )
    elif algorithm in ["Mixup", "OrgMixup"]:
        hparams["mixup_alpha"] = (0.2, 10 ** random_state.uniform(-1, -1))
    elif algorithm == "GroupDRO":
        hparams["groupdro_eta"] = (1e-2, 10 ** random_state.uniform(-3, -1))
    elif algorithm in ("MMD", "CORAL"):
        hparams["mmd_gamma"] = (1.0, 10 ** random_state.uniform(-1, 1))
    elif algorithm in ("MLDG", "SOMLDG"):
        hparams["mldg_beta"] = (1.0, 10 ** random_state.uniform(-1, 1))
    elif algorithm == "MTL":
        hparams["mtl_ema"] = (0.99, random_state.choice([0.5, 0.9, 0.99, 1.0]))
    elif algorithm == "VREx":
        hparams["vrex_lambda"] = (1e1, 10 ** random_state.uniform(-1, 5))
        hparams["vrex_penalty_anneal_iters"] = (
            500,
            int(10 ** random_state.uniform(0, 4)),
        )
    elif algorithm == "SAM":
        hparams["rho"] = (0.05, random_state.choice([0.01, 0.02, 0.05, 0.1]))
    elif algorithm == "CutMix":
        hparams["beta"] = (1.0, 1.0)
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        hparams["cutmix_prob"] = (1.0, 1.0)

    return hparams


def default_hparams(algorithm, dataset):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, dummy_random_state).items()}


def random_hparams(algorithm, dataset, seed):
    random_state = np.random.RandomState(seed)
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, random_state).items()}
