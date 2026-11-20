"""Run the fixed DPLCLIP protocol in dataset order and publish completed batches."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from domainbed.lib import dplclip_protocol as dpl

STEPS = 5001
FREQUENCY = 300
SEED = 1
RESULTS_BRANCH = "feature/dplclip-official-default-seed1-20261120"
ARCHIVE_FILES = ("config.json", "results.jsonl", "train.log", "best_summary.json")
PAPER = dict(PACS=97.3, VLCS=84.3, OfficeHome=84.2, TerraIncognita=52.6)
FIELDS = ("dataset", "target_domain", "seed", "lr", "batch_size", "steps",
          "best_step", "best_source_val", "test_out", "status")
SUMMARY = "dplclip_default_seed1_summary.csv"
COMPARISON = "dplclip_default_seed1_paper_comparison.json"
SECRET = re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|"
                    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|https?://[^\s/]+:[^\s/]+@")


def gpu_ready(gpu_uuid):
    while True:
        pids = subprocess.check_output(
            ["nvidia-smi", "-i", gpu_uuid, "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"], text=True,
        ).strip()
        if not pids:
            return
        print(f"Waiting for idle GPU {gpu_uuid}; compute processes: {pids}", flush=True)
        time.sleep(45)


def check_completion(path, dataset, target):
    best = json.loads((path / "best_summary.json").read_text())
    config = json.loads((path / "config.json").read_text())
    records = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
    steps = sorted(set(range(0, STEPS, FREQUENCY)) | {STEPS - 1})
    chosen = max(records, key=lambda record: record["source_val"])
    assert [r["step"] for r in records] == steps
    assert [config[key] for key in ("steps", "seed", "trial_seed")] == [STEPS, SEED, SEED]
    assert config["hparams"]["swad"] is None and config["lr_scheduler"] is None
    assert config["dataset"] == best["dataset"] == dataset and best["target_index"] == target
    assert (chosen["step"], chosen["source_val"], chosen["test_out"]) == (
        best["best_step"], best["best_source_val"], best["test_out"])
    assert best["restored_checkpoint_verified"]
    assert (path / "best_prompt_generator.pth").is_file()
    return {key: best[key] for key in ("best_step", "best_source_val", "test_out")}


def save_rows(root, rows):
    temporary = root / (SUMMARY + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / SUMMARY)
    report = {}
    for dataset, mean in PAPER.items():
        values = [r["test_out"] for r in rows if r["dataset"] == dataset and r["status"] == "complete"]
        actual = sum(values) / 4 * 100 if len(values) == 4 else None
        report[dataset] = {"completed": len(values), "mean_percent": actual, "paper_percent": mean,
                           "delta_pp": actual - mean if actual is not None else None}
    dpl.write_json(root / COMPARISON, report)


def run_fold(args, root, dataset, index, domain):
    gpu_ready(args.gpu)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONHASHSEED="1",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    command = [sys.executable, "-u", "train_all.py", "default_seed1", "--algorithm", "DPLCLIP",
               "--dataset", dataset, "--test_envs", str(index), "--seed", str(SEED),
               "--trial_seed", str(SEED), "--steps", str(STEPS), "--checkpoint_freq", str(FREQUENCY),
               "--data_dir", str(args.data_dir), "--dpl_clip_path", str(args.clip_path),
               "--num_workers", "8", "--eval_num_workers", "8", "--output_root", str(root)]
    print(f"Training {dataset} {domain}", flush=True)
    log = root / f".{dataset}.{domain}.console.log"
    with log.open("x") as handle:
        subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    (run,) = sorted((root / dataset / domain).glob("*_default_seed1"))
    statistics = check_completion(run, dataset, index)
    shutil.copyfile(log, run / "train.log")
    return run, statistics


def git(path, *argv, capture=False):
    return subprocess.run(["git", "-C", str(path), *argv], check=True, timeout=180,
                          capture_output=capture, text=True,
                          env=dict(os.environ, GIT_TERMINAL_PROMPT="0")).stdout


def publish(root, remote, dataset, runs, audit):
    if remote is None:
        return
    archive = root / ".github"
    if not (archive / ".git").is_dir():
        archive.mkdir(exist_ok=True)
        git(archive, "init", "-q", "-b", RESULTS_BRANCH)
        git(archive, "remote", "add", "origin", remote)
        git(archive, "config", "user.name", git(Path.cwd(), "config", "user.name", capture=True).strip())
        git(archive, "config", "user.email", git(Path.cwd(), "config", "user.email", capture=True).strip())

    destination = archive / "DCCL/DCCL/train_output/baselines/DPLCLIP/default_seed1"
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for relative in (SUMMARY, COMPARISON):
        paths[destination / relative] = root / relative
    for filename in ("data_audit.json", "smoke_console.log", "smoke_best_summary.json"):
        paths[destination / filename] = audit / filename
    for run in runs:
        for filename in ARCHIVE_FILES:
            paths[destination / run.relative_to(root) / filename] = run / filename

    for dest, src in paths.items():
        content = src.read_bytes()
        if SECRET.search(content):
            raise ValueError(f"Credential-like string in {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = content.replace(str(root).encode(), b"${DPL_OUTPUT_ROOT}")
        content = content.replace(str(Path.home()).encode(), b"${HOME}")
        dest.write_bytes(content)
    git(archive, "add", "--", str(destination.relative_to(archive)))
    if git(archive, "diff", "--cached", "--name-only", capture=True).strip():
        git(archive, "commit", "-m", f"Archive DPLCLIP seed 1 results through {dataset}")
    ref = f"refs/heads/{RESULTS_BRANCH}"
    git(archive, "push", "origin", f"HEAD:{ref}")
    local = git(archive, "rev-parse", "HEAD", capture=True).strip()
    remote_head = git(archive, "ls-remote", "origin", ref, capture=True).split()[0]
    assert local == remote_head
    print(f"Published {dataset} archive commit {local}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--clip-path", required=True, type=Path)
    parser.add_argument("--gpu", required=True, help="Idle CUDA GPU UUID")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--archive-remote", help="Private GitHub repository for experiment logs")
    args = parser.parse_args()
    smoke = json.loads((args.audit_dir / "smoke_best_summary.json").read_text())
    assert smoke["restored_checkpoint_verified"] and smoke["evaluations"] == 2
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = [dict(dataset=dataset, target_domain=domain, seed=SEED, lr=dpl.DEFAULTS["lr"],
                 batch_size=dpl.DEFAULTS["batch_size"], steps=STEPS, status="pending")
            for dataset, domains in dpl.DOMAINS.items() for domain in domains]
    with (root / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / SUMMARY).exists():
            raise FileExistsError("This output root already contains a DPL experiment")
        save_rows(root, rows)
        completed_runs = []
        publish_error = None
        target_index = 0
        for dataset, domains in dpl.DOMAINS.items():
            for index, domain in enumerate(domains):
                row = rows[target_index]
                target_index += 1
                row["status"] = "running"
                save_rows(root, rows)
                try:
                    run, statistics = run_fold(args, root, dataset, index, domain)
                    completed_runs.append(run)
                    row.update(statistics, status="complete")
                except (subprocess.CalledProcessError, OSError, ValueError, AssertionError) as exc:
                    row["status"] = "failed"
                    print(f"{dataset} {domain} failed: {exc}", flush=True)
                save_rows(root, rows)
            try:
                publish(root, args.archive_remote, dataset, completed_runs, args.audit_dir)
                publish_error = None
            except (subprocess.SubprocessError, OSError, ValueError, AssertionError) as exc:
                publish_error = str(exc)
                print(f"GitHub archive failed; local logs retained for retry: {exc}", flush=True)
        dpl.write_json(root / "queue_finished.json", {
            "completed": sum(row["status"] == "complete" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
            "publish_error": publish_error,
        })


if __name__ == "__main__":
    main()
