"""Publish allowlisted experiment records from a separate Git worktree."""
import fcntl
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from core import DATASET, NAME, ROOT

BRANCH = "results/" + NAME + "-" + DATASET.lower()
REMOTE = "https://github.com/r97993346-design/DCCL-3Source-DomainNet.git"
WORK = ROOT / ".github-sync"
TOP = {"best_target_iid.json", "validation_error.json", "preflight.json", "recovery_inventory.json", "negative_regression.json", "offline_regression.json", "recovery_records.jsonl", "validation_error.json", "repair_handoff.completed.json", "protocol.json", "snapshot.json", "fixed_trials.json", "search_space.json",
       "trials_summary.csv", "folds_summary.csv", "best_source_val.json", "best_iid.json",
       "best_test_out.json", "target_reached.json", "status.json", "controller.log", "pareto_front.csv", "pareto_front.json", "best_worst_fold_source_val.json", "decision_log.md", "parameter_sensitivity.json", "next_stage_analysis.md"}
TRIAL = {"config.json", "result.json", "results.jsonl", "learning_rates.jsonl",
         "console.log", "log.txt", "elapsed_seconds", "exit_code", "failure.json"}
SECRET = re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN [A-Z ]*PRIVATE KEY-----")


def git(*args, capture=False):
    return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "http.lowSpeedLimit=1024", "-c", "http.lowSpeedTime=20", "-C", str(WORK), *args],
                          env=dict(os.environ, GIT_TERMINAL_PROMPT="0"), check=True,
                          timeout=150, stdout=subprocess.PIPE if capture else None).stdout


def publish():
    WORK.mkdir(exist_ok=True)
    if not (WORK / ".git").exists():
        ref = git("ls-remote", REMOTE, f"refs/heads/{BRANCH}", capture=True)
        git("init", "-b", BRANCH)
        git("remote", "add", "origin", REMOTE)
        if ref:
            git("fetch", "origin", BRANCH)
            git("reset", "--hard", "FETCH_HEAD")
    for src in [*(ROOT / name for name in TOP), *(ROOT / DATASET / "trials").rglob("*")]:
        if not src.is_file() or src.is_symlink():
            continue
        rel = src.relative_to(ROOT)
        if len(rel.parts) > 1 and src.name not in TRIAL and not re.fullmatch(r"fold_[0-3]\.json", src.name):
            continue
        if src.stat().st_size > 64 * 1024**2 or SECRET.search(src.read_bytes()):
            print(f"Skipped {rel}", flush=True)
            continue
        dst = WORK / NAME / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    git("add", "--", NAME)
    if git("diff", "--cached", "--name-only", capture=True):
        git("-c", "user.name=CIPT automation", "-c", "user.email=cipt-automation@users.noreply.github.com",
            "commit", "-m", "Update multiobjective source-validation study records")
    try:
        git("push", "origin", f"HEAD:refs/heads/{BRANCH}")
    except subprocess.SubprocessError:
        from github_transport import publish as api_publish
        files = {str(path.relative_to(WORK)): path for path in (WORK / NAME).rglob("*") if path.is_file()}
        api_publish(BRANCH, files)


def main():
    with (ROOT / "sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        next_sync = 0
        while True:
            request = ROOT / "sync.request"
            folds = sorted((ROOT / DATASET / "trials").glob("trial_*/fold_*.json"))
            marker = (len(folds), request.stat().st_mtime_ns if request.exists() else 0)
            if marker != previous or time.monotonic() >= next_sync:
                try:
                    publish()
                except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                    print(f"GitHub sync failed ({type(exc).__name__}); retrying in 60 seconds", flush=True)
                    time.sleep(60)
                    continue
                previous, next_sync = marker, time.monotonic() + 1800
                status = ROOT / "status.json"
                if status.exists() and json.loads(status.read_text()).get("state") in {"target_reached", "space_exhausted", "stopped"}:
                    return
            time.sleep(5)


if __name__ == "__main__":
    main()
