import argparse
import collections
import copy
import itertools
import random
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from sconf import Config
from prettytable import PrettyTable

from domainbed.datasets import get_dataset
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.writers import get_writer
from domainbed.lib.logger import Logger
from domainbed.trainer import train
from domainbed.lib.cl_hparams import setup_alg_hparams

def main():
    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--data_dir", type=str, default="datadir/")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--algorithm", type=str, default="DCCL")
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and random_hparams).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of steps. Default is dataset-dependent."
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps. Default is dataset-dependent.",
    )
    parser.add_argument("--test_envs", type=int, nargs="+", default=None)  # sketch in PACS
    parser.add_argument("--source_envs", type=int, nargs="+", default=None, help="Source env indices for DomainNet (e.g., 0 1 2)")
    parser.add_argument("--target_env", type=int, default=None, help="Target env index for DomainNet (e.g., 5)")
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--model_save", default=None, type=int, help="Model save start step")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tb_freq", default=10)
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")
    parser.add_argument("--show", action="store_true", help="Show args and hparams w/o run")
    parser.add_argument(
        "--evalmode",
        default="fast",
        help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",
    )
    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")
    parser.add_argument("--model", type=str, default="resnet50", help="Backbone model architecture")
    parser.add_argument("--label_ratio", type=float, default=1.0, help="Ratio of labeled data to use")
    
    # Core DCCL hyperparameters (main tuning parameters)
    parser.add_argument("--l", type=float, default=1, help="Weight for contrastive loss between augmented views")
    parser.add_argument("--l_d", type=float, default=0.05, help="Weight for domain alignment regularization loss")
    parser.add_argument("--l_layer", type=float, default=1, help="Weight for layer-wise contrastive loss with pre-trained features")
    parser.add_argument("--t", type=float, default=0.1, help="Temperature parameter for contrastive loss")
    parser.add_argument("--t_pre", type=float, default=0.2, help="Temperature parameter for pre-trained feature contrastive loss")
    parser.add_argument("--n_layer", type=int, default=1, help="Number of layers in projection head")
    
    # Additional DCCL options (not in use, only for debugging and testing ideas)
    parser.add_argument("--sample_d", action="store_true", help="Enable domain-aware positive sampling")
    parser.add_argument("--re_w", action="store_true", help="Re-weight negative samples based on domain information")
    parser.add_argument("--sup", action="store_false", help="Use supervised contrastive learning (default: True)")
    parser.add_argument("--mix", type=float, default=0, help="Weight for sample mixing in dataset preprocessing")
    parser.add_argument("--aug", type=float, default=0, help="Probability of applying CutMix data augmentation")
    parser.add_argument("--two_ce", action="store_true", help="Use dual cross-entropy loss on both original and augmented views")
    parser.add_argument("--pos_mask", action="store_true", help="Apply positive mask in contrastive loss")
    parser.add_argument("--TN", action="store_true", help="Enable Transform Network for adversarial data augmentation")
    parser.add_argument("--lamda", type=float, default=5, help="Weight coefficient for Transform Network sparsity loss")
    parser.add_argument("--start_epoch", type=int, default=1000, help="Starting epoch for certain operations")
    parser.add_argument("--log", action="store_true", help="Enable detailed logging")
    args, left_argv = parser.parse_known_args()

    # setup hparams
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    hparams = setup_alg_hparams(hparams, args)
    keys = ["config.yaml"] + args.configs
    keys = [open(key, encoding="utf8") for key in keys]
    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv)

    # setup debug
    if args.debug:
        args.checkpoint_freq = 5
        args.steps = 10
        args.name += "_debug"

    timestamp = misc.timestamp()
    args.unique_name = f"{timestamp}_{args.name}"

    # path setup
    args.work_dir = Path(".")
    args.data_dir = Path(args.data_dir)

    args.out_root = args.work_dir / Path("train_output") / args.dataset
    args.out_dir = args.out_root / args.unique_name
    args.out_dir.mkdir(exist_ok=True, parents=True)

    writer = get_writer(args.out_root / "runs" / args.unique_name)
    logger = Logger.get(args.out_dir / "log.txt")
    if args.debug:
        logger.setLevel("DEBUG")
    cmd = " ".join(sys.argv)
    logger.info(f"Command :: {cmd}")

    logger.nofmt("Environment:")
    logger.nofmt("\tPython: {}".format(sys.version.split(" ")[0]))
    logger.nofmt("\tPyTorch: {}".format(torch.__version__))
    logger.nofmt("\tTorchvision: {}".format(torchvision.__version__))
    logger.nofmt("\tCUDA: {}".format(torch.version.cuda))
    logger.nofmt("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.nofmt("\tNumPy: {}".format(np.__version__))
    logger.nofmt("\tPIL: {}".format(PIL.__version__))

    # Different to DomainBed, we support CUDA only.
    assert torch.cuda.is_available(), "CUDA is not available"

    logger.nofmt("Args:")
    for k, v in sorted(vars(args).items()):
        logger.nofmt("\t{}: {}".format(k, v))

    logger.nofmt("HParams:")
    for line in hparams.dumps().split("\n"):
        logger.nofmt("\t" + line)

    if args.show:
        exit()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    # Dummy datasets for logging information.
    # Real dataset will be re-assigned in train function.
    # test_envs only decide transforms; simply set to zero.
    dataset, _in_splits, _out_splits = get_dataset([0], args, hparams)

    # print dataset information
    logger.nofmt("Dataset:")
    logger.nofmt(f"\t[{args.dataset}] #envs={len(dataset)}, #classes={dataset.num_classes}")
    for i, env_property in enumerate(dataset.environments):
        logger.nofmt(f"\tenv{i}: {env_property} (#{len(dataset[i])})")
    logger.nofmt("")

    if args.dataset == "DomainNet" and (args.source_envs is not None or args.target_env is not None):
        if args.source_envs is None or args.target_env is None:
            raise ValueError("For DomainNet custom split, --source_envs and --target_env must be provided together.")
        source_names = [dataset.environments[i] for i in args.source_envs]
        target_name = dataset.environments[args.target_env]
        logger.info(f"DomainNet custom split enabled: source_envs={args.source_envs} ({source_names}), target_env={args.target_env} ({target_name})")

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    logger.info(f"n_steps = {n_steps}")
    logger.info(f"checkpoint_freq = {checkpoint_freq}")

    org_n_steps = n_steps
    n_steps = (n_steps // checkpoint_freq) * checkpoint_freq + 1
    logger.info(f"n_steps is updated to {org_n_steps} => {n_steps} for checkpointing")

    if args.dataset == "DomainNet" and args.source_envs is not None and args.target_env is not None:
        args.test_envs = [[args.target_env]]
        logger.info(f"Target test envs = {args.test_envs} (DomainNet custom split)")
    else:
        if not args.test_envs:
            args.test_envs = [[te] for te in range(len(dataset))]
        else:
            args.test_envs = [[te] for te in args.test_envs]
        logger.info(f"Target test envs = {args.test_envs}")

    ###########################################################################
    # Run
    ###########################################################################
    all_records = []
    results = collections.defaultdict(list)

    combo_rows = []
    run_all_domainnet_triplets = (
        args.dataset == "DomainNet" and args.source_envs is None and args.target_env is None
    )
    if run_all_domainnet_triplets:
        env_ids = list(range(len(dataset)))
        env_names = list(dataset.environments)
        logger.info("Running all DomainNet 3-source->1-target combinations with ERM baseline.")
        for tgt in env_ids:
            source_candidates = [e for e in env_ids if e != tgt]
            for srcs in itertools.combinations(source_candidates, 3):
                run_args = copy.deepcopy(args)
                run_args.source_envs = list(srcs)
                run_args.target_env = tgt
                run_args.test_envs = [[tgt]]
                logger.info(
                    f"[Main] Combo source_envs={list(srcs)} ({[env_names[i] for i in srcs]}) -> "
                    f"target_env={tgt} ({env_names[tgt]})"
                )
                res, records = train(
                    [tgt], args=run_args, hparams=hparams, n_steps=n_steps,
                    checkpoint_freq=checkpoint_freq, logger=logger, writer=writer
                )
                erm_args = copy.deepcopy(run_args)
                erm_args.algorithm = "ERM"
                erm_hparams = hparams_registry.default_hparams("ERM", args.dataset)
                erm_hparams = setup_alg_hparams(erm_hparams, erm_args)
                erm_hparams = Config(default=erm_hparams)
                erm_res, _ = train(
                    [tgt], args=erm_args, hparams=erm_hparams, n_steps=n_steps,
                    checkpoint_freq=checkpoint_freq, logger=logger, writer=writer
                )
                key = "test_out"
                main_acc = float(res.get(key, np.mean(list(res.values()))))
                erm_acc = float(erm_res.get(key, np.mean(list(erm_res.values()))))
                rel_drop = (erm_acc - main_acc) / max(erm_acc, 1e-12)
                combo_rows.append((srcs, tgt, main_acc, erm_acc, rel_drop))
                all_records.append(records)
    else:
        if args.dataset == "DomainNet" and args.source_envs is not None and args.target_env is not None:
            args.test_envs = [[args.target_env]]
            logger.info(f"Target test envs = {args.test_envs} (DomainNet custom split)")
        for test_env in args.test_envs:
            res, records = train(
                test_env,
                args=args,
                hparams=hparams,
                n_steps=n_steps,
                checkpoint_freq=checkpoint_freq,
                logger=logger,
                writer=writer,
            )
            all_records.append(records)
            for k, v in res.items():
                results[k].append(v)

    # log summary table
    logger.info("=== Summary ===")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info("Unique name: %s" % args.unique_name)
    logger.info("Out path: %s" % args.out_dir)
    logger.info("Algorithm: %s" % args.algorithm)
    logger.info("Dataset: %s" % args.dataset)

    if run_all_domainnet_triplets:
        combo_table = PrettyTable(["Sources(3)", "Target", "Algo", "ERM", "RelDrop(vs ERM)"])
        for srcs, tgt, main_acc, erm_acc, rel_drop in combo_rows:
            src_names = ",".join([dataset.environments[i] for i in srcs])
            tgt_name = dataset.environments[tgt]
            combo_table.add_row(
                [src_names, tgt_name, f"{main_acc:.3%}", f"{erm_acc:.3%}", f"{rel_drop:.3%}"]
            )
        avg_algo = np.mean([r[2] for r in combo_rows]) if combo_rows else 0.0
        avg_erm = np.mean([r[3] for r in combo_rows]) if combo_rows else 0.0
        avg_rel = np.mean([r[4] for r in combo_rows]) if combo_rows else 0.0
        combo_table.add_row(["AVG", "ALL", f"{avg_algo:.3%}", f"{avg_erm:.3%}", f"{avg_rel:.3%}"])
        logger.nofmt(combo_table)
    else:
        max_metrics = max((len(v) for v in results.values()), default=0)
        env_headers = [f"env{i}" for i in range(max_metrics)]
        table = PrettyTable(["Selection"] + env_headers + ["Avg."])
        for key, values in results.items():
            avg = np.mean(values) if len(values) else 0.0
            row = [f"{acc:.3%}" for acc in values]
            row += ["-"] * (max_metrics - len(row))
            row.append(f"{avg:.3%}")
            table.add_row([key] + row)
        logger.nofmt(table)


if __name__ == "__main__":
    main()
