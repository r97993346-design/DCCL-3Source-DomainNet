import argparse
import collections
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
from domainbed.trainer import train
from domainbed.lib.cl_hparams import setup_alg_hparams
from domainbed.lib.dicm_hparams import finalize_dicm_hparams
from domainbed.lib.reproducibility import SEED_POLICY, seed_everything
from domainbed.lib.training_protocol import finalize_dccl_hparams, resolve_update_budget

def main():
    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--data_dir", type=str, default="datadir/")
    parser.add_argument("--output_root", type=Path, default=Path("train_output"),
                        help="Output root; dataset and timestamped run folders are appended")
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
    parser.add_argument("--tb_freq", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=None, help="Workers per train/eval loader; 0 runs in the main process")
    parser.add_argument("--step_count_mode", choices=("exact", "legacy"), default="exact",
                        help="exact: steps is the update budget; legacy: original DCCL checkpoint rounding")
    parser.add_argument("--isolate_eval_rng", action="store_true",
                        help="Preserve training RNG state during evaluation; default retains shared RNG")
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
    args.output_root = args.output_root.expanduser()
    args.seed_policy = SEED_POLICY
    if args.debug:
        args.checkpoint_freq = 5
        args.steps = 10
        args.name += "_debug"
    if args.tb_freq < 1 or (args.num_workers is not None and args.num_workers < 0):
        parser.error("tb_freq must be positive and num_workers must be nonnegative")
    dataset_class = datasets.get_dataset_class(args.dataset)
    args.requested_steps = dataset_class.N_STEPS if args.steps is None else args.steps
    args.resolved_checkpoint_freq = dataset_class.CHECKPOINT_FREQ if args.checkpoint_freq is None else args.checkpoint_freq
    args.resolved_steps = resolve_update_budget(args.requested_steps, args.resolved_checkpoint_freq, args.step_count_mode)

    # setup hparams
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    hparams = setup_alg_hparams(hparams, args)
    keys = ["config.yaml"] + args.configs
    keys = [open(key, encoding="utf8") for key in keys]
    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv)

    if args.algorithm == "DICM":
        finalize_dicm_hparams(hparams, args, sys.argv[1:])
    elif args.algorithm == "DCCL":
        finalize_dccl_hparams(hparams, args, sys.argv[1:])
    # A0 configuration review also needs neither CUDA, data nor output folders.
    if args.show:
        print(f"Algorithm: {args.algorithm}")
        print(f"Seed policy: {args.seed_policy}; seed={args.seed}; trial_seed={args.trial_seed}")
        print(f"Step count: {args.step_count_mode}; requested_steps={args.requested_steps}; max_updates={args.resolved_steps}; checkpoint_freq={args.resolved_checkpoint_freq}")
        print(f"Evaluation RNG: {'isolated' if args.isolate_eval_rng else 'shared'}; num_workers={dataset_class.N_WORKERS if args.num_workers is None else args.num_workers}")
        print(f"Output root: {args.output_root / args.dataset}")
        print(hparams.dumps())
        return

    timestamp = misc.timestamp()
    args.unique_name = f"{timestamp}_{args.name}"

    # path setup
    args.work_dir = Path(".")
    args.data_dir = Path(args.data_dir)

    args.out_root = args.work_dir / args.output_root / args.dataset
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

    # Seed metadata setup too; trainer.train resets again before EACH target.
    seed_everything(args.seed, args.deterministic)

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

    n_steps = args.resolved_steps
    checkpoint_freq = args.resolved_checkpoint_freq
    logger.info(f"n_steps = {n_steps}")
    logger.info(f"checkpoint_freq = {checkpoint_freq}")

    logger.info(f"Step count mode={args.step_count_mode}, requested={args.requested_steps}, max_updates={n_steps}")

    if not args.test_envs:
        args.test_envs = [[te] for te in range(len(dataset))]
    else:
        args.test_envs = [[te] for te in args.test_envs]
    targets = [env[0] for env in args.test_envs]
    if len(set(targets)) != len(targets) or any(te < 0 or te >= len(dataset) for te in targets):
        raise ValueError("test_envs must be unique valid target-domain IDs")
    logger.info(f"Target test envs = {args.test_envs}")

    ###########################################################################
    # Run
    ###########################################################################
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

    table = PrettyTable(["Selection"] + [dataset.environments[te] for te in targets] + ["Avg."])
    for key, row in results.items():
        row = [f"{acc:.3%}" for acc in [*row, np.mean(row)]]
        
        table.add_row([key] + row)
    logger.nofmt(table)


if __name__ == "__main__":
    main()

