#!/usr/bin/env python3
"""Audit whether CIRL/ICR and RISE branch functionality is present.

The user-requested source branches are external to this container in some runs.
This script makes the audit reproducible: it prints the required git commands,
reports missing refs/remotes, and checks for the concrete capabilities expected
from the ICR/Fourier and RISE-guided DCCL work.
"""

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

CAPABILITY_CHECKS = {
    "ICR / Fourier intervention": [
        ("domainbed/modules/cirl.py", "fourier_intervention"),
        ("domainbed/algorithms/algorithms.py", "loss_icr_fourier"),
        ("train_all.py", "--lambda_icr"),
    ],
    "Reliability reweighting": [
        ("domainbed/modules/cirl.py", "reliability_matrix"),
        ("domainbed/algorithms/algorithms.py", "reliability_matrix=reliability_matrix"),
        ("domainbed/algorithms/algorithms.py", "mask = mask * reliability_matrix"),
    ],
    "RISE CLIP teacher": [
        ("domainbed/modules/rise.py", "class RISETeacher"),
        ("domainbed/modules/rise.py", "clip.load"),
        ("domainbed/modules/rise.py", "requires_grad = False"),
    ],
    "KD loss": [
        ("domainbed/modules/rise.py", "compute_kd_loss"),
        ("domainbed/algorithms/algorithms.py", "loss_kd"),
        ("train_all.py", "--lambda_kd"),
    ],
    "AD semantic alignment loss": [
        ("domainbed/modules/rise.py", "compute_ad_loss"),
        ("domainbed/algorithms/algorithms.py", "loss_ad"),
        ("train_all.py", "--lambda_ad"),
    ],
    "rise80 prompt templates": [
        ("domainbed/modules/rise.py", "RISE80_PROMPTS"),
        ("domainbed/modules/rise.py", "rise80"),
        ("train_all.py", "--rise_prompt_mode"),
    ],
    "Training entry parameters": [
        ("train_all.py", "--use_cirl"),
        ("train_all.py", "--use_rise"),
        ("train_all.py", "--rise_clip_download_root"),
    ],
    "hparams/config registration": [
        ("domainbed/hparams_registry.py", "use_cirl"),
        ("domainbed/hparams_registry.py", "use_rise"),
        ("domainbed/lib/cl_hparams.py", "rise_kd_temperature"),
    ],
    "DomainNet run script": [
        ("scripts/run_domainnet_cirl_rise.py", "dccl_cirl_rise"),
        ("scripts/run_domainnet_cirl_rise.py", "--source_envs"),
        ("scripts/run_domainnet_cirl_rise.py", "--target"),
    ],
}


def run(cmd):
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode, completed.stdout.strip()


def has_ref(ref):
    code, _ = run(["git", "rev-parse", "--verify", ref])
    return code == 0


def check_capability(name, checks):
    missing = []
    for relpath, needle in checks:
        path = REPO_ROOT / relpath
        if not path.exists() or needle not in path.read_text(errors="ignore"):
            missing.append(f"{relpath}: {needle}")
    return missing


def main():
    parser = argparse.ArgumentParser(description="Audit CIRL/ICR and RISE branch merge status")
    parser.add_argument("--icr-ref", default="feat-icr-dccl-fourier-reliability")
    parser.add_argument("--rise-ref", default="feat-rise-guided-dccl")
    args = parser.parse_args()

    commands = [
        ["git", "branch", "--show-current"],
        ["git", "log", "--oneline", "--decorate", "--graph", "--all", "--max-count=40"],
        ["git", "diff", "--stat", f"HEAD..{args.icr_ref}"],
        ["git", "diff", "--stat", f"HEAD..{args.rise_ref}"],
    ]
    print("# Required git command output")
    for cmd in commands:
        code, output = run(cmd)
        print(f"\n$ {' '.join(cmd)}")
        print(output if output else "<no output>")
        if code != 0:
            print(f"[exit={code}]")

    print("\n# Reference availability")
    for ref in [args.icr_ref, args.rise_ref, "codex/conduct-code-audit-for-icr-dccl", "codex/implement-rise-guided-dccl-method"]:
        print(f"{ref}: {'present' if has_ref(ref) else 'missing'}")
    code, remotes = run(["git", "remote", "-v"])
    print("remotes:", remotes if remotes else "<none>")

    print("\n# Capability checks")
    failed = False
    for name, checks in CAPABILITY_CHECKS.items():
        missing = check_capability(name, checks)
        if missing:
            failed = True
            print(f"FAIL {name}")
            for item in missing:
                print(f"  missing {item}")
        else:
            print(f"PASS {name}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
