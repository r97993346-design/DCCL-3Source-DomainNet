#!/usr/bin/env python3
"""Run normal DomainNet 3-source -> 1-target DCCL/RISE jobs.

This launcher is intentionally for the full/normal DomainNet layout
(<data_dir>/domain_net/{clip,info,paint,quick,real,sketch}). It does not create
or expect a reduced/sub10 dataset.
"""

import argparse
import itertools
import os
from pathlib import Path
import subprocess
import sys


DOMAINNET_ENVS = ["clip", "info", "paint", "quick", "real", "sketch"]
VARIANTS = ["baseline", "rise_proto", "rise_kd", "rise_kd_proto"]
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent


def validate_env_id(value, name):
    if value < 0 or value >= len(DOMAINNET_ENVS):
        raise ValueError(f"{name} must be in [0, {len(DOMAINNET_ENVS) - 1}], got {value}")


def validate_full_domainnet(data_dir):
    domainnet_root = Path(data_dir) / "domain_net"
    if not domainnet_root.is_dir():
        raise FileNotFoundError(
            f"Normal DomainNet root not found: {domainnet_root}. "
            "Expected <data_dir>/domain_net/{clip,info,paint,quick,real,sketch}."
        )
    missing = [env for env in DOMAINNET_ENVS if not (domainnet_root / env).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"DomainNet root {domainnet_root} is missing environment folders: {missing}. "
            "This launcher is for the normal full DomainNet 3-source -> 1-target setup, not a reduced/sub10 set."
        )
    return domainnet_root


def variant_args(variant, args):
    if variant == "baseline":
        return []
    clip_args = ["--rise_clip_model_name", args.rise_clip_model_name]
    if args.rise_clip_download_root is not None:
        clip_args.extend(["--rise_clip_download_root", str(Path(args.rise_clip_download_root).resolve())])
    common_rise_args = [
        "--use_rise",
        *clip_args,
        "--rise_prompt_mode", args.rise_prompt_mode,
    ]
    proto_args = [
        "--use_rise_proto",
        "--rise_proto_weight", str(args.rise_proto_weight),
        "--rise_projection_dim", str(args.rise_projection_dim),
    ]
    kd_args = [
        "--use_rise_kd",
        "--rise_kd_weight", str(args.rise_kd_weight),
        "--rise_kd_temperature", str(args.rise_kd_temperature),
    ]
    if variant == "rise_proto":
        return common_rise_args + proto_args
    if variant == "rise_kd":
        return common_rise_args + kd_args
    if variant == "rise_kd_proto":
        return common_rise_args + kd_args + proto_args
    raise ValueError(f"Unknown variant: {variant}")


def run_command(cmd, gpu, dry_run):
    print("[CMD]", f"(cd {TRAIN_DIR} && {' '.join(cmd)})", flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(cmd, check=True, env=env, cwd=TRAIN_DIR)


def build_commands(args):
    data_dir = str(Path(args.data_dir).resolve())
    validate_full_domainnet(data_dir)
    variants = VARIANTS if args.variant == "all" else [args.variant]

    if args.source_envs is not None:
        if args.target_env is None:
            raise ValueError("--target_env is required when --source_envs is provided.")
        if len(args.source_envs) != 3:
            raise ValueError(f"Normal DomainNet run requires exactly 3 source envs, got {args.source_envs}.")
        for env_id in args.source_envs:
            validate_env_id(env_id, "source env id")
        validate_env_id(args.target_env, "--target_env")
        if args.target_env in args.source_envs:
            raise ValueError("--source_envs and --target_env must be disjoint.")
        combos = [(tuple(args.source_envs), args.target_env)]
    else:
        if args.target_env is None:
            targets = range(len(DOMAINNET_ENVS))
        else:
            validate_env_id(args.target_env, "--target_env")
            targets = [args.target_env]
        combos = []
        env_ids = range(len(DOMAINNET_ENVS))
        for target in targets:
            source_candidates = [env_id for env_id in env_ids if env_id != target]
            combos.extend((sources, target) for sources in itertools.combinations(source_candidates, 3))

    commands = []
    for sources, target in combos:
        source_tag = "".join(str(env_id) for env_id in sources)
        combo_tag = f"s{source_tag}_t{target}"
        for variant in variants:
            run_name = f"{args.exp_prefix}_{combo_tag}_{variant}"
            cmd = [
                sys.executable,
                "train_all.py",
                run_name,
                "--dataset", "DomainNet",
                "--data_dir", data_dir,
                "--algorithm", "DCCL",
                "--source_envs", *(str(env_id) for env_id in sources),
                "--target_env", str(target),
            ]
            cmd.extend(variant_args(variant, args))
            cmd.extend(args.extra_args)
            commands.append(cmd)
    return commands


def main():
    parser = argparse.ArgumentParser(
        description="Run normal full DomainNet 3-source -> 1-target DCCL/RISE jobs."
    )
    parser.add_argument("--data_dir", required=True, help="Data root containing domain_net/{clip,info,paint,quick,real,sketch}")
    parser.add_argument("--source_envs", type=int, nargs=3, default=None, help="Exactly three source env ids. Omit to sweep combinations.")
    parser.add_argument("--target_env", type=int, default=None, help="Target env id. Omit with --source_envs omitted to sweep all targets.")
    parser.add_argument(
        "--variant",
        choices=VARIANTS + ["all"],
        default="all",
        help=(
            "Which variant to run: baseline, rise_proto (AD/text prototype only), "
            "rise_kd (KD only), rise_kd_proto (AD+KD), or all."
        ),
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES value for launched jobs.")
    parser.add_argument("--exp_prefix", default="domainnet_3source", help="Run-name prefix.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--rise_clip_model_name", default="ViT-B/32", help="CLIP model name or local checkpoint path.")
    parser.add_argument("--rise_clip_download_root", default=None, help="Local directory/cache for CLIP weights on offline servers.")
    parser.add_argument("--rise_kd_weight", type=float, default=0.5)
    parser.add_argument("--rise_proto_weight", type=float, default=0.1)
    parser.add_argument("--rise_kd_temperature", type=float, default=2.0)
    parser.add_argument("--rise_prompt_mode", choices=["simple", "multi", "domain_invariant", "rise80"], default="multi")
    parser.add_argument("--rise_projection_dim", type=int, default=512)
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[], help="Additional args appended to train_all.py commands.")
    args = parser.parse_args()

    commands = build_commands(args)
    print(
        f"[INFO] normal DomainNet launcher: commands={len(commands)}, "
        f"variant={args.variant}, dry_run={args.dry_run}",
        flush=True,
    )
    for cmd in commands:
        run_command(cmd, args.gpu, args.dry_run)


if __name__ == "__main__":
    main()
