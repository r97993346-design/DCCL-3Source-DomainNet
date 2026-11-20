import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from sconf import Config
from prettytable import PrettyTable

from domainbed.datasets import get_dataset, datasets
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.writers import get_writer
from domainbed.lib.logger import Logger
from domainbed.trainer import train as train_fold
from experiment_archive import Experiment
from domainbed.lib.cl_hparams import setup_alg_hparams
from domainbed.lib.dicm_hparams import finalize_dicm_hparams

def main():
    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--output_root", type=Path, default=Path("train_output"))
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
    parser.add_argument("--stage", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--stage_parent", type=Path, default=None)
    parser.add_argument("--stage_resume", type=Path, default=None,
                        help="Restore model, optimizer, RNG and SWAD; DataLoader stream is not exact")
    parser.add_argument("--stage_save_steps", type=int, nargs="*", default=None)
    parser.add_argument("--pacs_causal_checkpoints", action="store_true")
    parser.add_argument("--pacs_causal_resume", type=Path)
    parser.add_argument("--pacs_matched_protocol", type=Path)
    args, left_argv = parser.parse_known_args()
    if args.pacs_causal_checkpoints or args.pacs_causal_resume:
        if (not args.pacs_causal_checkpoints or args.dataset != "PACS"
                or args.algorithm != "DICM" or args.stage is not None
                or args.debug or args.test_envs is None or len(args.test_envs) != 1):
            parser.error("PACS checkpointing requires a single end-to-end DICM PACS fold")

    # setup hparams
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    hparams = setup_alg_hparams(hparams, args)
    keys = ["config.yaml"] + args.configs
    keys = [open(key, encoding="utf8") for key in keys]
    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv)
    if args.algorithm == "DICM":
        if args.stage == 2 and str(hparams["dicm_ablation"]).upper() == "A5":
            hparams["dicm_stage2_gt"] = True
        elif "dicm_stage2_gt" in hparams:
            hparams["dicm_stage2_gt"] = False
        finalize_dicm_hparams(hparams)

    if args.stage is not None:
        from domainbed.lib.stagewise import validate_stage
        if args.algorithm != "DICM" or not args.test_envs or len(args.test_envs) != 1:
            raise ValueError("Stagewise training requires DICM and exactly one held-out --test_envs")
        if not args.steps or not args.checkpoint_freq:
            raise ValueError("Stagewise training requires explicit --steps and --checkpoint_freq")
        if args.stage_resume is not None and args.stage_parent is not None:
            raise ValueError("Specify a stage parent or a stage resume, not both")
        parent = args.stage_parent or (args.stage_resume if args.stage else None)
        validate_stage(args.stage, hparams, parent, dataset=args.dataset)

    if (args.pacs_matched_protocol or hparams.get("dicm_fixed_reference", False)) and not args.pacs_causal_checkpoints:
        parser.error("Matched PACS runs and fixed reference require --pacs_causal_checkpoints")

    # setup debug
    if args.debug:
        args.checkpoint_freq = 5
        args.steps = 10
        args.name += "_debug"

    if args.show:
        dataset_class = datasets.get_dataset_class(args.dataset)
        requested_steps = args.steps or dataset_class.N_STEPS
        checkpoint_freq = args.checkpoint_freq or dataset_class.CHECKPOINT_FREQ
        max_updates = requested_steps if args.stage is not None else (requested_steps // checkpoint_freq) * checkpoint_freq + 1
        print(f"Algorithm: {args.algorithm}")
        print(f"Seed policy: per_run; seed={args.seed}; trial_seed={args.trial_seed}")
        print(f"Requested steps={requested_steps}; max_updates={max_updates}; checkpoint_freq={checkpoint_freq}")
        print(hparams.dumps())
        return

    timestamp = misc.timestamp()
    args.unique_name = f"{timestamp}_{args.name}"

    # path setup
    args.work_dir = Path(".")
    args.data_dir = Path(args.data_dir)

    args.out_root = args.work_dir / args.output_root.expanduser() / args.dataset
    args.out_dir = args.out_root / args.unique_name
    if args.pacs_causal_resume:
        latest = args.pacs_causal_resume.resolve(strict=True)
        if (latest.name != "latest.pth" or latest.parent.name != "checkpoints"
                or latest.parent.parent.parent != args.out_root.resolve()
                or not latest.parent.parent.name.endswith("_" + args.name)
                or (latest.parent.parent / ".github-archive").exists()):
            parser.error("Resume requires latest.pth of this incomplete PACS fold")
        args.out_dir = latest.parent.parent
        args.unique_name = args.out_dir.name
    else:
        args.out_dir.mkdir(exist_ok=False, parents=True)

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

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    logger.info(f"n_steps = {n_steps}")
    logger.info(f"checkpoint_freq = {checkpoint_freq}")

    org_n_steps = n_steps
    n_steps = n_steps if args.stage is not None else (n_steps // checkpoint_freq) * checkpoint_freq + 1
    logger.info(f"n_steps is updated to {org_n_steps} => {n_steps} for checkpointing")

    if not args.test_envs:
        args.test_envs = [[te] for te in range(len(dataset))]
    else:
        args.test_envs = [[te] for te in args.test_envs]
    logger.info(f"Target test envs = {args.test_envs}")

    ###########################################################################
    # Run
    ###########################################################################
    archive = Experiment("DICM", args, dataset, logger)
    train = archive.wrap(train_fold)
    all_records = []
    results = collections.defaultdict(list)

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

    target_names = [dataset.environments[te] for (te,) in args.test_envs]
    table = PrettyTable(["Selection"] + target_names + ["Avg."])
    for key, row in results.items():
        row.append(np.mean(row))
        row = [f"{acc:.3%}" for acc in row]
        
        table.add_row([key] + row)
    logger.nofmt(table)

    archive.finish()


if __name__ == "__main__":
    main()
