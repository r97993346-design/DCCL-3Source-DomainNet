#!/usr/bin/env python3
"""Run DomainNet DCCL ablations for CIRL/ICR and RISE.

This script keeps the existing source_envs/target_env path intact while making
it easy to launch the required ablations on a fixed target domain.
"""

import argparse
import itertools
import os
import subprocess
import sys


ABLATIONS = {
    "dccl": [],
    "dccl_cirl": [
        "--use_cirl",
        "--lambda_cirl", "1.0",
        "--cirl_use_fourier_reliability",
        "--lambda_icr", "1.0",
    ],
    "dccl_cirl_no_fourier": [
        "--use_cirl",
        "--lambda_cirl", "1.0",
        "--lambda_icr", "0.0",
    ],
    "dccl_rise": [
        "--use_rise",
        "--lambda_kd", "1.0",
        "--lambda_ad", "1.0",
    ],
    "dccl_rise_kd_only": [
        "--use_rise",
        "--lambda_kd", "1.0",
        "--lambda_ad", "0.0",
    ],
    "dccl_rise_ad_only": [
        "--use_rise",
        "--lambda_kd", "0.0",
        "--lambda_ad", "1.0",
    ],
    "dccl_cirl_rise": [
        "--use_cirl",
        "--lambda_cirl", "1.0",
        "--cirl_use_fourier_reliability",
        "--lambda_icr", "1.0",
        "--use_rise",
        "--lambda_kd", "1.0",
        "--lambda_ad", "1.0",
    ],
}


def run(cmd, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run DomainNet DCCL+CIRL/ICR+RISE ablations")
    parser.add_argument("--target", type=int, required=True, help="Target env id in [0,5]")
    parser.add_argument("--source_envs", type=int, nargs="+", default=None, help="Explicit source env ids; defaults to all combinations")
    parser.add_argument("--source_count", type=int, default=3, choices=[3, 5], help="Number of sources when --source_envs is omitted")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--clip_cache", type=str, default=None, help="RISE CLIP cache path")
    parser.add_argument("--exp_prefix", type=str, default="domainnet_cirl_rise")
    parser.add_argument("--ablations", nargs="+", default=["dccl", "dccl_cirl", "dccl_rise", "dccl_cirl_rise"], choices=sorted(ABLATIONS))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--checkpoint_freq", type=int, default=500)
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    envs = [0, 1, 2, 3, 4, 5]
    if args.target not in envs:
        raise ValueError(f"--target must be in [0,5], got {args.target}")
    if args.source_envs is None:
        source_sets = itertools.combinations([env for env in envs if env != args.target], args.source_count)
    else:
        if args.target in args.source_envs:
            raise ValueError("--source_envs and --target must be disjoint")
        source_sets = [tuple(args.source_envs)]

    for source_envs in source_sets:
        src_tag = "".join(map(str, source_envs))
        for ablation in args.ablations:
            name = f"{args.exp_prefix}_{ablation}_s{src_tag}_t{args.target}"
            cmd = [
                sys.executable, "train_all.py", name,
                "--dataset", "DomainNet",
                "--algorithm", "DCCL",
                "--data_dir", args.data_dir,
                "--source_envs", *map(str, source_envs),
                "--target_env", str(args.target),
                "--steps", str(args.steps),
                "--checkpoint_freq", str(args.checkpoint_freq),
                *ABLATIONS[ablation],
            ]
            if "rise" in ablation:
                if args.clip_cache is None:
                    raise ValueError("RISE ablations require --clip_cache")
                cmd += [
                    "--rise_prompt_mode", "rise80",
                    "--rise_clip_model_name", "ViT-B/32",
                    "--rise_clip_download_root", args.clip_cache,
                    "--rise_kd_temperature", "2.0",
                ]
            cmd += args.extra_args
            run(cmd, args.gpu)


if __name__ == "__main__":
    main()
