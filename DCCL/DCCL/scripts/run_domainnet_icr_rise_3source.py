#!/usr/bin/env python3
"""Example launcher for DomainNet 3-source ICR + RISE DCCL experiments."""

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Run DomainNet ICR + RISE DCCL example.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--clip_cache", required=True)
    parser.add_argument("--target_env", type=int, default=4)
    parser.add_argument("--source_envs", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--checkpoint_freq", type=int, default=500)
    parser.add_argument("--cuda_visible_devices", default="0")
    args = parser.parse_args()

    cmd = [
        "python",
        "train_all.py",
        f"ICR_RISE_DCCL_sub10_t{args.target_env}_s0",
        "--dataset",
        "DomainNet",
        "--algorithm",
        "DCCL",
        "--data_dir",
        args.data_dir,
        "--source_envs",
        *[str(env) for env in args.source_envs],
        "--target_env",
        str(args.target_env),
        "--use_fourier_intervention",
        "true",
        "--use_rise",
        "--use_rise_kd",
        "--use_rise_proto",
        "--rise_clip_model_name",
        "ViT-B/32",
        "--rise_clip_download_root",
        args.clip_cache,
        "--rise_prompt_mode",
        "rise80",
        "--rise_kd_weight",
        "0.5",
        "--rise_proto_weight",
        "0.1",
        "--rise_kd_temperature",
        "2.0",
        "--rise_projection_dim",
        "512",
        "--steps",
        str(args.steps),
        "--checkpoint_freq",
        str(args.checkpoint_freq),
    ]
    env = {"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices}
    subprocess.run(cmd, check=True, env={**env})


if __name__ == "__main__":
    main()
