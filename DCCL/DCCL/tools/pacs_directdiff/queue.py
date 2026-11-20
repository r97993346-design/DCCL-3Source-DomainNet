"""Independent DirectDiff16 queues, checked against the historical PACS splits."""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.pacs_directdiff.common import (
    DATA, DOMAINS, GROUPS, PROJECT, ROOT, atomic_json, folds, query, say,
)


MEMORY_MIB = 30720
DISK_GIB = 24


def tmux_name(group):
    return f"directdiff16_{group}"


def wait_gpu(gpu):
    while True:
        if shutil.disk_usage(ROOT).free < DISK_GIB * 1024**3:
            raise RuntimeError(f"Less than {DISK_GIB} GiB free for checkpoints and feature analysis")
        free_text, uuid = query("-i", str(gpu), "--query-gpu=memory.free,uuid").split(",")
        free = int(free_text)
        conflicting_pid = None
        if free > MEMORY_MIB:
            for pid in query("-i", str(gpu), "--query-compute-apps=pid").splitlines():
                try:
                    args = (Path("/proc") / pid.strip() / "cmdline").read_bytes().split(b"\0")
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                if any(args[i:i + 2] == [b"--algorithm", b"DICM"] for i in range(len(args) - 1)):
                    conflicting_pid = pid
                    break
            if conflicting_pid is None:
                return uuid.strip(), free
        note = f"DICM PID {conflicting_pid} active" if conflicting_pid else f"need >{MEMORY_MIB} MiB"
        say(f"GPU{gpu} waiting: {free} MiB free, {note}")
        time.sleep(60)


def protocol(index):
    return ROOT / "configs" / f"A16_env{index}.json"


def command(group, index):
    config = Path(__file__).parent
    files = [str(PROJECT / "tools" / "pacs_projection" / "A16.yaml"), str(config / "direct.yaml")]
    if group == "Gate":
        files.append(str(config / "gate.yaml"))
    return [
        sys.executable, "-B", "-u", "train_all.py", f"{group}_env{index}", *files,
        "--algorithm", "DICM", "--dataset", "PACS", "--test_envs", str(index),
        "--steps", "5001", "--checkpoint_freq", "100", "--seed", "0", "--trial_seed", "0",
        "--deterministic", "--holdout_fraction", "0.2", "--label_ratio", "1",
        "--model", "resnet50", "--data_dir", str(DATA), "--output_root", str(ROOT / group),
        "--pacs_causal_checkpoints", "--pacs_matched_protocol", str(protocol(index)),
    ]


def check_hparams(group, original, actual):
    expected = dict(original, dicm_direct_difference=True, dicm_direction_gate=group == "Gate",
                    dicm_residual_projection=False, dicm_gate_warmup=300, dicm_gate_interval=100)
    if actual != expected:
        different = {k: [expected.get(k), actual.get(k)] for k in actual.keys() | expected.keys()
                     if actual.get(k) != expected.get(k) or (k in actual) != (k in expected)}
        raise ValueError(f"This fold differs from the frozen A16 recipe: {different}")


def completed(folder, group, index, commit):
    comparison = json.loads(protocol(index).read_text())
    outbox = folder / ".github-archive"
    manifest = json.loads((outbox / "manifest.json").read_text())
    fold, = json.loads((outbox / "evaluation.json").read_text())
    assert manifest["code"]["commit"] == commit and fold["target"] == [DOMAINS[index]]
    assert fold["checkpoint_freq"] == 100 and fold["steps"] == 5001
    args = fold["args"]
    assert args["seed"] == args["trial_seed"] == 0 and args["deterministic"]
    assert args["holdout_fraction"] == .2 and args["label_ratio"] == 1
    assert fold["swad_evaluation"] is not None
    check_hparams(group, comparison["hparams"], fold["hparams"])
    for path in ("iid_best", "swad", "latest"):
        assert (folder / "checkpoints" / f"{path}.pth").is_file()
    assert (folder / "sample_manifest.json").is_file()
    log = (folder / "log.txt").read_text()
    assert "A16 split and initial Backbone/Classifier verified" in log
    assert all(f"Reload verified for {name}.pth" in log for name in ("iid_best", "swad"))
    if group == "Gate":
        assert (folder / "swad_gate.json").is_file()
        assert list((folder / "gate_updates").glob("*.json"))
    return fold


def run(group):
    gpu = GROUPS[group]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True).strip():
        raise RuntimeError("Run DirectDiff16 from a clean, pinned worktree")
    preflight = ROOT / "logs" / "preflight.passed.json"
    if not preflight.is_file() or json.loads(preflight.read_text())["commit"] != commit:
        raise RuntimeError("DirectDiff16 training commit has no matching test receipt")
    running = ROOT / "logs" / f"{group}.running.json"
    with (ROOT / f"{group}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, domain in enumerate(DOMAINS):
            name = f"{group}_env{index}"
            found = folds(group, index)
            if len(found) > 1:
                raise RuntimeError(f"Duplicate fold directory for {name}")
            if found and (found[0] / ".github-archive").is_dir():
                completed(found[0], group, index, commit)
                say(name, "already complete")
                continue
            cmd = command(group, index)
            if found:
                latest = found[0] / "checkpoints" / "latest.pth"
                if not latest.is_file():
                    raise RuntimeError(f"Existing incomplete fold lacks recovery state: {found[0]}")
                cmd.extend(["--pacs_causal_resume", str(latest)])
            uuid, free = wait_gpu(gpu)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
            console = ROOT / "logs" / f"{name}.log"
            with console.open("a", encoding="utf-8", buffering=1) as stream:
                say(name, domain, f"GPU{gpu} {free} MiB free")
                with subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT) as child:
                    atomic_json(running, dict(group=group, gpu=gpu, pid=child.pid, target=domain,
                                              index=index, commit=commit, console=str(console)))
                    code = child.wait()
                running.unlink(missing_ok=True)
                if code != 0:
                    raise RuntimeError(f"{name} stopped with exit {code}. See {console}")
            path, = folds(group, index)
            fold = completed(path, group, index, commit)
            say(name, domain, f"IID {fold['selection']['iid']:.4%}",
                f"SWAD {fold['selection']['SWAD']:.4%}",
                "converged", fold["swad_evaluation"]["converged"])
        atomic_json(ROOT / "logs" / f"{group}.complete.json",
                    dict(group=group, commit=commit, folds=len(DOMAINS)))
        say(group, "completed all four PACS folds")


if __name__ == "__main__":
    group, = sys.argv[1:]
    try:
        run(group)
    except Exception as error:
        atomic_json(ROOT / "logs" / f"{group}.failure.json",
                    dict(group=group, error=type(error).__name__, reason=str(error)))
        raise
