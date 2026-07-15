#!/usr/bin/env python3
import argparse
import itertools
import os
import subprocess
import sys


def run(cmd, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run DomainNet fixed-target source-combos, DCCL+ERM")
    parser.add_argument("--target", type=int, required=True, help="Target env id in [0,5]")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id for all jobs")
    parser.add_argument("--data_dir", type=str, required=True, help="DomainNet data root")
    parser.add_argument("--exp_prefix", type=str, default="exp-domainnet-target")
    parser.add_argument("--erm_baseline", choices=["weak", "matched"], default="weak", help="ERM command mode: weak adds --weak_erm (ImageNet pretrained, no SWAD); matched keeps ERM hparams matched to the main run.")
    parser.add_argument("--source_count", type=int, default=3, choices=[3, 5], help="Number of source domains per fixed-target combo. Default 3 preserves the original sweep; use 5 for full 5-source DomainNet runs.")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    if args.target < 0 or args.target > 5:
        raise ValueError(f"--target must be in [0,5], got {args.target}")

    envs = [0, 1, 2, 3, 4, 5]
    sources = [e for e in envs if e != args.target]
    combos = list(itertools.combinations(sources, args.source_count))
    print(f"[INFO] target={args.target}, gpu={args.gpu}, source_count={args.source_count}, combos={len(combos)}")

    for srcs in combos:
        combo_tag = f"s{''.join(map(str, srcs))}_t{args.target}"
        base = [
            sys.executable,
            "train_all.py",
            "--dataset", "DomainNet",
            "--data_dir", args.data_dir,
            "--source_envs", *map(str, srcs),
            "--target_env", str(args.target),
        ] + args.extra_args

        run(base[:2] + [f"{args.exp_prefix}_{combo_tag}_dccl", "--algorithm", "DCCL"] + base[2:], args.gpu)
        erm_extra = ["--weak_erm"] if args.erm_baseline == "weak" else []
        run(base[:2] + [f"{args.exp_prefix}_{combo_tag}_erm", "--algorithm", "ERM"] + base[2:] + erm_extra, args.gpu)

    print(f"[DONE] target={args.target} finished all {len(combos)} combos with DCCL+ERM.")


if __name__ == "__main__":
    main()
