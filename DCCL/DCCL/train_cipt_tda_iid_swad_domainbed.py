#!/usr/bin/env python3
"""Original CIPT modules + candidate-consistent TDA + source-only IID/SWAD.

Reuse the existing DomainBed splits and upstream training loop unchanged.
An optimizer post-step hook collects dense trainable-weight SWAD snapshots.
Target partitions are evaluated only after all selections have been fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from cipt_iid_swad import (
    IIDSelector, SWADTracker, evaluate_loader, evaluate_sources,
    load_trainable_state, preserve_rng, save_checkpoint, trainable_state,
    write_json, write_summary,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["PACS", "VLCS", "OfficeHome", "TerraIncognita", "DomainNet"], default="PACS")
    p.add_argument("--data-dir", default="")
    p.add_argument("--clip-path", default="")
    p.add_argument("--backbone", default="ViT-B/16")
    p.add_argument("--test-envs", nargs="+", type=int)
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trial-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.0025)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--gamma", type=float, default=5.0)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--n-ctx", type=int, default=16)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-num-workers", type=int, default=0)
    p.add_argument("--model-variant", choices=["consistent", "official"], default="consistent")
    p.add_argument("--no-swad", action="store_true")
    p.add_argument("--swad-n-converge", type=int, default=3)
    p.add_argument("--swad-n-tolerance", type=int, default=6)
    p.add_argument("--swad-tolerance-ratio", type=float, default=0.3)
    p.add_argument("--eval-every", type=int, default=1, help="Source validation every N epochs when --checkpoint-freq=0")
    p.add_argument("--checkpoint-freq", type=int, default=0, help="Source validation every N optimizer steps; 0 uses --eval-every epochs")
    p.add_argument("--output-root", default="train_output/ablation/cipt_tda_iid_swad")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--summarize-only", action="store_true", help="Rebuild tables from saved per-target metrics; no data/CLIP needed")
    args = p.parse_args(argv)
    for key in ("epochs", "batch_size", "eval_batch_size", "k", "n_ctx", "num_heads", "eval_every"):
        if getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    if min(args.num_workers, args.eval_num_workers, args.checkpoint_freq) < 0:
        p.error("Worker counts and checkpoint-freq must be nonnegative")
    if not 0 < args.holdout_fraction < 1:
        p.error("holdout-fraction must lie between 0 and 1")
    if args.lr <= 0 or args.weight_decay < 0:
        p.error("lr must be positive and weight-decay nonnegative")
    if min(args.swad_n_converge, args.swad_n_tolerance) < 2 or args.swad_tolerance_ratio < 0:
        p.error("SWAD windows must be >=2 and tolerance ratio nonnegative")
    if args.test_envs is not None and (len(set(args.test_envs)) != len(args.test_envs) or min(args.test_envs) < 0):
        p.error("test-envs must contain distinct nonnegative ids")
    if not args.summarize_only:
        if not args.data_dir:
            p.error("--data-dir is required for training")
        if args.clip_path and not Path(args.clip_path).is_file():
            p.error(f"clip-path does not exist: {args.clip_path}")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_target(args, target, reference, official):
    build_cipt, cosine_scheduler, _evaluate, make_optimizer, train_one_epoch = official
    dataset, names, train, source_val, target_in, target_out, sizes = reference.make_splits(args, target)
    env_name = dataset.environments[target]
    directory = Path(args.output_root) / args.dataset / f"target_{target}_{env_name}" / f"seed_{args.seed}_trial_{args.trial_seed}"
    if directory.exists() and any(directory.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{directory} contains results; use a new output root or explicit --overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_cipt(
        names, backbone=args.clip_path or args.backbone, device=device,
        n_ctx=args.n_ctx, ctx_init="a photo of a", num_diverse_templates=args.k,
        num_heads=args.num_heads, sample_templates=True,
    )
    # The averaging wrapper holds parameters only; mutable train-time buffers
    # would need additional handling. Current CIPT heads use Linear/LayerNorm.
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training
           for name in ("causal_adapter", "spurious_adapter", "diversity_augmentation", "prompt_learner")
           for m in getattr(model, name, torch.nn.Identity()).modules()):
        raise ValueError("This runner supports the original CIPT Linear/LayerNorm heads, without train-time BatchNorm")
    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    schedule = cosine_scheduler(optimizer, epochs=args.epochs)
    train_loader = reference.loader(train, args.batch_size, args.num_workers, shuffle=True)
    source_names = list(sizes["source_val"])
    if len(source_names) != len(source_val.datasets):
        raise RuntimeError("Source validation domain/split mismatch")
    source_loaders = {
        name: reference.loader(split, args.eval_batch_size, args.eval_num_workers)
        for name, split in zip(source_names, source_val.datasets)
    }
    iid = IIDSelector()
    swad = None if args.no_swad else SWADTracker(
        model, device, args.swad_n_converge, args.swad_n_tolerance,
        args.swad_tolerance_ratio,
    )
    config = {
        **vars(args), "test_env": target, "environment": env_name,
        "upstream_commit": reference.UPSTREAM_COMMIT,
        "split": "DCCL domainbed.datasets.get_dataset",
        "source_validation_reduction": "domain_macro_mean",
        "iid_selection": "max_source_validation_accuracy_earliest_tie",
        "swad_selection": None if swad is None else "LossValley_source_validation_CE",
        "swad_average_frequency": "every_optimizer_step",
        "inference": "TDA", "primary_target_metric": "target_out",
        "classes": names, "sizes": sizes,
    }
    write_json(directory / "config.json", config)
    print(f"Target {target} ({env_name}): {sizes}; output={directory}", flush=True)
    history, epoch_history = [], []
    step, last_evaluated_step, epoch_index = 0, -1, 0
    results_path = directory / "results.jsonl"
    results_path.write_text("", encoding="utf-8")

    def checkpoint():
        nonlocal last_evaluated_step
        if step == last_evaluated_step:
            return
        validation = evaluate_sources(model, source_loaders, device)
        if iid.consider(model, validation, epoch_index, step):
            save_checkpoint(directory / "iid_best_trainable.pth", iid.state, {**config, **iid.metadata})
        if swad is not None:
            swad.checkpoint(validation)
        row = {
            "step": step, "epoch": step / len(train_loader),
            "epoch_index": epoch_index, "lr": optimizer.param_groups[0]["lr"],
            "train_out": validation["accuracy"], "tr_outloss": validation["loss"],
            "source_val": validation["accuracy"],
            "source_val_pooled": validation["pooled_accuracy"],
            "source_val_per_domain": validation["per_domain"],
            "iid_best_step": iid.metadata["step"],
            "iid_best_source_val": iid.metadata["source_val"],
            "swad_dead_valley": None if swad is None else swad.valley.dead_valley,
        }
        history.append(row)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        write_json(directory / "history.json", history)
        last_evaluated_step = step
        print("VALID " + json.dumps(row), flush=True)

    def after_optimizer_step(_optimizer, _args, _kwargs):
        nonlocal step
        step += 1
        if swad is not None:
            swad.update(step)
        if args.checkpoint_freq and step % args.checkpoint_freq == 0:
            checkpoint()

    hook = optimizer.register_step_post_hook(after_optimizer_step)
    try:
        for epoch_index in range(1, args.epochs + 1):
            losses = train_one_epoch(
                model, train_loader, optimizer, device, beta=args.beta, gamma=args.gamma,
            )
            schedule.step()
            if not args.checkpoint_freq and epoch_index % args.eval_every == 0:
                checkpoint()
            row = {"epoch": epoch_index, "step": step, "lr": optimizer.param_groups[0]["lr"], **losses}
            epoch_history.append(row)
            write_json(directory / "epoch_losses.json", epoch_history)
            print("TRAIN " + json.dumps(row), flush=True)
        checkpoint()  # Always include final weights and the trailing SWAD segment.
    finally:
        hook.remove()

    last_state = trainable_state(model)
    last_metadata = {"selection": "last", "epoch": args.epochs, "step": step}
    save_checkpoint(directory / "final_trainable.pth", last_state, {**config, **last_metadata})
    states = {"iid": (iid.state, iid.metadata), "last": (last_state, last_metadata)}
    swad_metadata = None
    if swad is not None:
        swad_state, swad_metadata = swad.finalize()
        states["SWAD"] = (swad_state, {"selection": "SWAD", **swad_metadata})
        save_checkpoint(directory / "swad_trainable.pth", swad_state, {**config, **states["SWAD"][1]})

    # All state selection and checkpoint writes are complete before target use.
    target_loaders = {
        "target_out": reference.loader(target_out, args.eval_batch_size, args.eval_num_workers),
        "target_in": reference.loader(target_in, args.eval_batch_size, args.eval_num_workers),
    }
    selections = {}
    try:
        for selection, (state, metadata) in states.items():
            load_trainable_state(model, state)
            validation = evaluate_sources(model, source_loaders, device)
            with preserve_rng():
                targets = {key: evaluate_loader(model, item, device) for key, item in target_loaders.items()}
            selections[selection] = {
                **metadata,
                "source_val": validation["accuracy"], "source_val_loss": validation["loss"],
                "source_val_per_domain": validation["per_domain"],
                "target_out": targets["target_out"]["accuracy"],
                "target_in": targets["target_in"]["accuracy"],
                "target_metrics": targets,
            }
    finally:
        load_trainable_state(model, last_state)
    final = {
        "target": env_name, "test_env": target, "model_variant": args.model_variant,
        "primary_target_metric": "target_out", "selections": selections, "swad": swad_metadata,
    }
    write_json(directory / "metrics.json", final)
    print("FINAL " + json.dumps(final), flush=True)
    del model, optimizer, swad
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final


def summarize_saved(args):
    base = Path(args.output_root) / args.dataset
    results = []
    for path in base.glob(f"target_*/seed_{args.seed}_trial_{args.trial_seed}/metrics.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if args.test_envs is None or value["test_env"] in args.test_envs:
            if value["model_variant"] != args.model_variant:
                raise ValueError(f"Model variant mismatch in {path}")
            results.append(value)
    if args.test_envs is None:
        results.sort(key=lambda r: r["test_env"])
    else:
        by_id = {r["test_env"]: r for r in results}
        missing = set(args.test_envs) - by_id.keys()
        if missing:
            raise FileNotFoundError(f"Missing completed targets: {sorted(missing)}")
        results = [by_id[i] for i in args.test_envs]
    return write_summary(base, args.dataset, args.model_variant, results, args.seed, args.trial_seed)


def main(argv=None):
    args = parse_args(argv)
    if args.summarize_only:
        summarize_saved(args)
        return
    import train_cipt_official_domainbed as reference
    official = reference.load_official_api()
    if args.model_variant == "consistent":
        from cipt_tda_consistent import build_consistent_cipt
        official = (build_consistent_cipt, *official[1:])
    from domainbed.datasets.datasets import get_dataset_class
    count = len(get_dataset_class(args.dataset).ENVIRONMENTS)
    ids = list(range(count)) if args.test_envs is None else args.test_envs
    if any(i >= count for i in ids):
        raise ValueError(f"Invalid target id; valid ids are {list(range(count))}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    results = []
    for target in ids:
        seed_everything(args.seed)
        results.append(run_target(args, target, reference, official))
        write_summary(
            Path(args.output_root) / args.dataset, args.dataset, args.model_variant,
            results, args.seed, args.trial_seed,
        )


if __name__ == "__main__":
    main()
