"""Train each PACS target in its own process and synchronize completed folds."""
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_archive import publish


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path("/home/hooasia/lgg/projects/dicm-pacs-causal-validation-20261119")
DATA = Path("/home/hooasia/lgg/data/repro_dccl_data")
DOMAINS = ("art_painting", "cartoon", "photo", "sketch")
GROUPS = {"P0": 0, "P1": 1}
MIN_GPU_MIB = 20480
MIN_DISK = 12 * 1024**3


def say(*message):
    print(datetime.now().astimezone().isoformat(timespec="seconds"), *message, flush=True)


def atomic_json(path, payload):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def query(*args):
    return subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                                   text=True).strip()


def resource_check(gpu):
    free, uuid = (x.strip() for x in query("-i", str(gpu), "--query-gpu=memory.free,uuid").split(","))
    free = int(free)
    if free <= MIN_GPU_MIB:
        return None, f"{free} MiB free (requires >{MIN_GPU_MIB})"
    if shutil.disk_usage(ROOT).free < MIN_DISK:
        raise RuntimeError("Output disk has less than 12 GiB free")
    for line in query("-i", str(gpu), "--query-compute-apps=pid").splitlines():
        if not line.strip().isdigit():
            continue
        try:
            args = (Path("/proc") / line.strip() / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(args[i:i+2] == [b"--algorithm", b"DICM"] for i in range(len(args)-1)):
            return None, f"existing DICM GPU process {line.strip()}"
    return uuid, f"{free} MiB free"


def wait_for_gpu(gpu):
    while True:
        uuid, status = resource_check(gpu)
        if uuid:
            return uuid, status
        say(f"GPU{gpu} waiting:", status)
        time.sleep(60)


def folds(group, index):
    return sorted((ROOT / group / "PACS").glob(f"*_{group}_env{index}"))


def completed(folder, group, index, commit):
    outbox = folder / ".github-archive"
    manifest = json.loads((outbox / "manifest.json").read_text(encoding="utf-8"))
    fold, = json.loads((outbox / "evaluation.json").read_text(encoding="utf-8"))
    assert manifest["code"]["commit"] == commit and fold["target"] == [DOMAINS[index]]
    assert fold["steps"] == 5001 and fold["checkpoint_freq"] == 100
    assert fold["hparams"]["dicm_ablation"] == {"P0": "A1", "P1": "A2"}[group]
    assert fold["args"]["seed"] == fold["args"]["trial_seed"] == 0
    assert fold["swad_evaluation"] is not None
    for kind in ("iid_best", "swad", "latest"):
        assert (folder / "checkpoints" / (kind + ".pth")).is_file()
    log = (folder / "log.txt").read_text(encoding="utf-8")
    assert all(f"Reload verified for {kind}.pth" in log for kind in ("iid_best", "swad"))
    return fold


def train(group):
    gpu = GROUPS[group]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True).strip():
        raise RuntimeError("Training worktree must match its pinned commit")
    if not all((DATA / "PACS" / name).is_dir() for name in DOMAINS):
        raise RuntimeError("Expected four PACS domains are unavailable")
    with (ROOT / f"{group}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, domain in enumerate(DOMAINS):
            name = f"{group}_env{index}"
            existing = folds(group, index)
            if len(existing) > 1:
                raise RuntimeError(f"Duplicate output for {name}")
            if existing and (existing[0] / ".github-archive").is_dir():
                completed(existing[0], group, index, commit)
                say(name, "already finished")
                continue
            cmd = [sys.executable, "-B", "-u", "train_all.py", name,
                   str(Path(__file__).with_name(group + ".yaml")), "--algorithm", "DICM",
                   "--dataset", "PACS", "--test_envs", str(index), "--steps", "5001",
                   "--checkpoint_freq", "100", "--seed", "0", "--trial_seed", "0",
                   "--deterministic", "--holdout_fraction", "0.2", "--label_ratio", "1",
                   "--model", "resnet50", "--data_dir", str(DATA),
                   "--output_root", str(ROOT / group), "--pacs_causal_checkpoints"]
            if existing:
                latest = existing[0] / "checkpoints/latest.pth"
                if not latest.is_file():
                    raise RuntimeError(f"Inspect incomplete fold without a recovery checkpoint: {existing[0]}")
                cmd.extend(["--pacs_causal_resume", str(latest)])
            uuid, status = wait_for_gpu(gpu)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
            console = ROOT / "logs" / f"{name}.log"
            with console.open("a", encoding="utf-8", buffering=1) as handle:
                say(name, domain, "GPU" + str(gpu), status, "log", console)
                child = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                atomic_json(ROOT / "logs" / f"{group}.running.json",
                            dict(pid=child.pid, gpu=gpu, target=domain, fold=index,
                                 commit=commit, command=cmd, console=str(console)))
                result = child.wait()
            if result:
                raise RuntimeError(f"{name} stopped with code {result}. See {console}")
            outdir, = folds(group, index)
            row = completed(outdir, group, index, commit)
            say(name, domain, f"IID {row['selection']['iid']:.4%}, SWAD {row['selection']['SWAD']:.4%}",
                "converged", row["swad_evaluation"]["converged"])
            (ROOT / "logs" / f"{group}.running.json").unlink(missing_ok=True)
        atomic_json(ROOT / "logs" / f"{group}.complete.json", dict(commit=commit, folds=4))
        say(group, "all 4 PACS folds complete")


def sync():
    with (ROOT / "sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            synchronized = 0
            for group in GROUPS:
                for index in range(len(DOMAINS)):
                    for folder in folds(group, index):
                        outbox = folder / ".github-archive"
                        if not outbox.is_dir():
                            continue
                        receipt = folder / "archive_receipt.json"
                        try:
                            if not receipt.is_file():
                                commit = publish(outbox)
                                atomic_json(receipt, dict(commit=commit))
                                say("GitHub verified", group, DOMAINS[index], commit)
                            synchronized += 1
                        except (OSError, subprocess.SubprocessError) as err:
                            say("GitHub retry scheduled", folder.name, type(err).__name__)
            if synchronized == 8:
                say("All eight fold bundles synchronized")
                return
            time.sleep(60)


if __name__ == "__main__":
    task, = sys.argv[1:]
    if task == "sync":
        sync()
    else:
        train(task)
