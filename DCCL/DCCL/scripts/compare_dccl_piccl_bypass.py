#!/usr/bin/env python3
"""Deterministic one-batch DCCL vs PICCL(use_piccl=false) bypass check."""
import argparse
import copy
import sys

import torch

from domainbed.algorithms.algorithms import DCCL
from domainbed.algorithms.piccl import PICCL


def default_hparams(use_piccl):
    return {
        "data_augmentation": True, "val_augment": False, "resnet18": False,
        "resnet_dropout": 0.0, "class_balanced": False, "optimizer": "adam",
        "freeze_bn": True, "pretrained": False, "lr": 1e-3, "batch_size": 4,
        "weight_decay": 0.0, "t": 0.1, "t_pre": 0.2, "l": 1,
        "l_d": 0.01, "l_layer": 1, "n_layer": 1, "sup": True,
        "two_ce": False, "sample_d": False, "re_w": False, "pos_mask": False,
        "mix": 0, "aug": 0, "TN": False, "lamda": 5, "model": "resnet50",
        "start_epoch": 1000, "log": False, "use_piccl": use_piccl,
    }


def first_param_diff(a, b):
    b_params = dict(b.named_parameters())
    for name, pa in a.named_parameters():
        pb = b_params.get(name)
        if pb is None:
            return f"missing parameter in PICCL-off: {name}"
        if not torch.equal(pa, pb):
            return f"parameter differs: {name}, max_abs={(pa-pb).abs().max().item():.6g}"
    extra = set(b_params) - {n for n, _ in a.named_parameters()}
    if extra:
        return f"extra parameters in PICCL-off: {sorted(extra)[:3]}"
    return None


def assert_equal_tensor(name, a, b):
    if not torch.equal(a, b):
        print(f"FAIL first tensor difference: {name}; max_abs={(a-b).abs().max().item():.6g}")
        return False
    print(f"OK {name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-domains", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dccl = DCCL((3, args.image_size, args.image_size), args.num_classes, args.num_domains, default_hparams(False))
    torch.manual_seed(args.seed)
    piccl = PICCL((3, args.image_size, args.image_size), args.num_classes, args.num_domains, default_hparams("false"))
    dccl.eval(); piccl.eval()

    diff = first_param_diff(dccl, piccl)
    if diff:
        print(f"FAIL first initialization difference: {diff}")
        return 1
    print("OK initialization parameters")

    g = torch.Generator().manual_seed(args.seed + 1)
    xs = [torch.randn(args.batch_size, 3, args.image_size, args.image_size, generator=g) for _ in range(args.num_domains)]
    x2 = [x + 0.01 for x in xs]
    ys = [torch.arange(args.batch_size) % args.num_classes for _ in range(args.num_domains)]
    ds = [torch.full((args.batch_size,), i, dtype=torch.long) for i in range(args.num_domains)]
    all_x = torch.cat(xs)

    with torch.no_grad():
        z_d = dccl.featurizer(all_x)
        z_p = piccl.featurizer(all_x)
        if not assert_equal_tensor("backbone feature", z_d, z_p): return 1
        log_d = dccl.predict(all_x)
        log_p = piccl.predict(all_x)
        if not assert_equal_tensor("logits", log_d, log_p): return 1

    kwargs = {"x_2": x2, "d": ds, "d_2": ds, "step": 0}
    out_d = dccl.update(copy.deepcopy(xs), copy.deepcopy(ys), **copy.deepcopy(kwargs))
    out_p = piccl.update(copy.deepcopy(xs), copy.deepcopy(ys), **copy.deepcopy(kwargs))
    for key in ["ce_loss", "sup_cl_loss", "pre_cl_loss", "loss"]:
        if out_d.get(key) != out_p.get(key):
            print(f"FAIL first loss difference: {key}: DCCL={out_d.get(key)} PICCL-off={out_p.get(key)}")
            return 1
        print(f"OK {key}={out_d[key]}")

    diff = first_param_diff(dccl, piccl)
    if diff:
        print(f"FAIL first post-step parameter difference: {diff}")
        return 1
    print("OK first optimizer.step parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
