#!/usr/bin/env python3
"""Unmodified upstream CIPT on this repository's DomainBed data splits.

The CIPT implementation is loaded from third_party/CIPT (pinned Git submodule).
No CIPTDCCL, RD, MCC, contrastive learning, SWAD or modified adapter code is used.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from domainbed.datasets import get_dataset
from domainbed.hparams_registry import default_hparams

UPSTREAM_COMMIT = "a805d878acc7d79778d1ec1c1e4d73ba6aff334b"
OFFICIAL_DEFAULTS = dict(
    epochs=30, batch_size=64, lr=0.0025, beta=4.0, gamma=5.0,
    k=4, n_ctx=16, num_heads=8, weight_decay=0.0,
)


class ImageLabelPairs(Dataset):
    """Adapt the DomainBed dict sample to the upstream (image, label) interface."""

    def __init__(self, split):
        self.split = split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        item = self.split[index]
        return item["x"], item["y"]


def load_official_api():
    root = Path(__file__).resolve().parent / "third_party" / "CIPT"
    if not (root / "cipt" / "model.py").is_file():
        raise RuntimeError(
            "Missing upstream CIPT submodule. From the repository root run: "
            "git submodule update --init --recursive"
        )
    sys.path.insert(0, str(root))
    from cipt import build_cipt, cosine_scheduler, evaluate, make_optimizer, train_one_epoch
    return build_cipt, cosine_scheduler, evaluate, make_optimizer, train_one_epoch


def make_splits(args, target):
    # Call the exact loader used in the parent DCCL branch: its split_dataset
    # uses seed_hash(trial_seed, environment_id) and holdout_fraction.
    hparams = default_hparams("CIPTDCCL", args.dataset)
    hparams.update(
        label_ratio=1.0, sample_d=False, mix=0.0,
        cipt_enabled=True, cipt_pure=True, cipt_use_aug_view=False,
        data_augmentation=True, val_augment=False, class_balanced=False,
    )
    split_args = argparse.Namespace(
        dataset=args.dataset, data_dir=Path(args.data_dir),
        source_envs=None, target_env=None,
        trial_seed=args.trial_seed, holdout_fraction=args.holdout_fraction,
    )
    # The original repository selects official CLIP preprocessing via this
    # algorithm class name; the model itself is always the unmodified upstream.
    clip_transform_marker = type("CIPTDCCL", (), {})
    dataset, in_splits, out_splits = get_dataset(
        [target], split_args, hparams, algorithm_class=clip_transform_marker
    )
    sources = [i for i in range(len(dataset)) if i != target]
    classes = list(dataset[0].classes)
    for env in dataset:
        if list(env.classes) != classes:
            raise RuntimeError("Inconsistent class order between DomainBed domains")
    train = ConcatDataset([ImageLabelPairs(in_splits[i][0]) for i in sources])
    source_val = ConcatDataset([ImageLabelPairs(out_splits[i][0]) for i in sources])
    target_in = ImageLabelPairs(in_splits[target][0])
    target_out = ImageLabelPairs(out_splits[target][0])
    sizes = {
        "source_train": {dataset.environments[i]: len(in_splits[i][0]) for i in sources},
        "source_val": {dataset.environments[i]: len(out_splits[i][0]) for i in sources},
        "target_in": len(target_in), "target_out": len(target_out),
    }
    if not train or not source_val or not target_in or not target_out:
        raise RuntimeError("Empty DomainBed partition; inspect dataset and holdout_fraction")
    return dataset, [name.replace("_", " ") for name in classes], train, source_val, target_in, target_out, sizes


def loader(dataset, batch_size, workers, shuffle=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        drop_last=False, persistent_workers=workers > 0,
    )


def save_trainable_checkpoint(path, model, metadata):
    # Frozen OpenAI CLIP weights come from --clip-path or --backbone at reload.
    # Avoid duplicating the ~hundreds-of-MB frozen encoder for every target.
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable
    }
    torch.save({"trainable_state_dict": state, "metadata": metadata}, path)


def run_target(args, target, official):
    build_cipt, cosine_scheduler, evaluate, make_optimizer, train_one_epoch = official
    dataset, names, train, source_val, target_in, target_out, sizes = make_splits(args, target)
    env_name = dataset.environments[target]
    out_dir = Path(args.output_root) / args.dataset / f"target_{target}_{env_name}" / f"seed_{args.seed}_trial_{args.trial_seed}"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} already contains results; pass --overwrite deliberately")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Avoid tuning on the held-out domain. CIPT's upstream code trains all 30
    # epochs with Adam + cosine; there is deliberately no SWAD/early stopping.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_or_path = args.clip_path or args.backbone
    model, _official_preprocess = build_cipt(
        names, backbone=backbone_or_path, device=device,
        n_ctx=args.n_ctx, ctx_init="a photo of a",
        num_diverse_templates=args.k, num_heads=args.num_heads,
        sample_templates=True,
    )
    opt = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    schedule = cosine_scheduler(opt, epochs=args.epochs)
    train_loader = loader(train, args.batch_size, args.num_workers, shuffle=True)
    val_loader = loader(source_val, args.eval_batch_size, args.num_workers)
    target_in_loader = loader(target_in, args.eval_batch_size, args.num_workers)
    target_out_loader = loader(target_out, args.eval_batch_size, args.num_workers)

    config = {
        "upstream": "https://github.com/ckghostwj/CIPT",
        "upstream_commit": UPSTREAM_COMMIT, "split": "DCCL domainbed.datasets.get_dataset",
        "dataset": args.dataset, "environment": env_name, "test_env": target,
        "seed": args.seed, "trial_seed": args.trial_seed,
        "holdout_fraction": args.holdout_fraction,
        "no_few_shot_sampling": True, "no_swad": True,
        "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
        "weight_decay": args.weight_decay, "beta": args.beta, "gamma": args.gamma,
        "k": args.k, "n_ctx": args.n_ctx, "num_heads": args.num_heads,
        "backbone_or_path": str(backbone_or_path),
        "source_val_for_model_selection": False,
        "sizes": sizes, "classes": names,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Target {target} ({env_name}): {sizes}; output={out_dir}", flush=True)
    history = []
    for epoch in range(args.epochs):
        losses = train_one_epoch(
            model, train_loader, opt, device, beta=args.beta, gamma=args.gamma
        )
        schedule.step()
        row = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], **losses}
        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            row["source_val"] = evaluate(model, val_loader, device, use_tda=True)["accuracy"]
        history.append(row)
        print(json.dumps(row), flush=True)

    # The target domain is untouched throughout training and model selection.
    final = {
        "target": env_name,
        "target_in": evaluate(model, target_in_loader, device, use_tda=True)["accuracy"],
        "target_out": evaluate(model, target_out_loader, device, use_tda=True)["accuracy"],
        "source_val": evaluate(model, val_loader, device, use_tda=True)["accuracy"],
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    save_trainable_checkpoint(out_dir / "final_trainable.pth", model, config)
    print(f"FINAL {json.dumps(final)}", flush=True)
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final


def parse_args():
    p = argparse.ArgumentParser(description="Official CIPT + DomainBed data splits (full data)")
    p.add_argument("--dataset", choices=["PACS", "VLCS", "OfficeHome", "TerraIncognita", "DomainNet"], default="PACS")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--clip-path", default="", help="Local ViT-B/16 .pt (preferred on offline servers)")
    p.add_argument("--backbone", default="ViT-B/16")
    p.add_argument("--test-envs", nargs="*", type=int, default=None)
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    p.add_argument("--trial-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=OFFICIAL_DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=OFFICIAL_DEFAULTS["batch_size"])
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=OFFICIAL_DEFAULTS["lr"])
    p.add_argument("--beta", type=float, default=OFFICIAL_DEFAULTS["beta"])
    p.add_argument("--gamma", type=float, default=OFFICIAL_DEFAULTS["gamma"])
    p.add_argument("--k", type=int, default=OFFICIAL_DEFAULTS["k"])
    p.add_argument("--n-ctx", type=int, default=OFFICIAL_DEFAULTS["n_ctx"])
    p.add_argument("--num-heads", type=int, default=OFFICIAL_DEFAULTS["num_heads"])
    p.add_argument("--weight-decay", type=float, default=OFFICIAL_DEFAULTS["weight_decay"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--output-root", default="train_output/cipt_official_domainbed")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if not (0 < args.holdout_fraction < 1):
        p.error("holdout-fraction must be strictly between 0 and 1")
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0 or args.k <= 0:
        p.error("epochs, batch-size, k must be positive and num-workers nonnegative")
    if args.clip_path and not Path(args.clip_path).is_file():
        p.error(f"clip-path does not exist: {args.clip_path}")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    official = load_official_api()
    # Verify domain list before any training; no test domain is ever sampled.
    from domainbed.datasets.datasets import get_dataset_class
    envs = get_dataset_class(args.dataset).ENVIRONMENTS
    ids = list(range(len(envs))) if args.test_envs is None else args.test_envs
    if len(set(ids)) != len(ids) or any(i < 0 or i >= len(envs) for i in ids):
        raise ValueError(f"Invalid/duplicate target environment ids: {ids}; valid={list(range(len(envs)))}")
    results = []
    for target in ids:
        # Re-seed so each target run is independent of earlier training runs.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        results.append(run_target(args, target, official))
    summary = {
        "dataset": args.dataset, "targets": results,
        "target_in_mean": float(np.mean([x["target_in"] for x in results])),
        "target_out_mean": float(np.mean([x["target_out"] for x in results])),
    }
    base = Path(args.output_root) / args.dataset
    base.mkdir(parents=True, exist_ok=True)
    (base / f"summary_targets_{'-'.join(str(i) for i in ids)}_seed_{args.seed}_trial_{args.trial_seed}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
