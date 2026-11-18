"""Matched reliability experiments initialized from historical effective configurations."""
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import experiment_archive as archive

ROOT = Path("/home/hooasia/lgg/projects/dicm-reliability-paired-20261118")
DATA = Path("/home/hooasia/lgg/data/repro_dccl_data")
SOURCE = subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True).strip()
HIST = Path("/home/hooasia/lgg/projects/repro_dccl_3domain/DCCL/DCCL/train_output/dicm_residual_seed0_20261112")
RUNS = {"PACS": "261113_11-23-29_PACS_full_06", "VLCS": "261113_20-16-13_VLCS_domain_precl_20"}
DOMAINS = {"PACS": ("art_painting", "cartoon", "photo", "sketch"),
           "VLCS": ("CALTECH", "LABELME", "PASCAL", "SUN")}
HISTORICAL_SWAD = {"PACS": (0.908481, 0.843284, 0.984281, 0.841285),
                   "VLCS": (0.988516, 0.645176, 0.794150, 0.779893)}
GAMMAS = {"B0": 0.0, "B1": 0.5, "B2": 1.0}
PROTOCOL = ("Matched per-target seed 0; historical runs used one process with a single run seed. "
            "Historical source commit is unverified. All paired B0/B1/B2 runs use this pinned source.")


def say(text):
    print(datetime.now().astimezone().isoformat(timespec="seconds"), text, flush=True)


def git(*args):
    return subprocess.check_output(["git", "-C", str(PROJECT), *args], text=True).strip()


def verify_source():
    if git("rev-parse", "HEAD") != SOURCE or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Pinned experiment code changed")


def historical_hparams(dataset):
    path = HIST / dataset / RUNS[dataset] / "results.jsonl"
    with path.open(encoding="utf8") as stream:
        first = json.loads(next(stream))
        hp = first["hparams"]
        for line in stream:
            if json.loads(line)["hparams"] != hp:
                raise ValueError("Historical hparams differ across folds")
    if hp["dicm_ablation"] != "A4" or hp["l_d"] != 0 or hp["optimizer"] != "adam":
        raise ValueError("Unexpected historical GT or optimizer configuration")
    return hp


def make_config(dataset, gamma):
    hp = historical_hparams(dataset)
    return {**hp, "dicm_reliable_contrast_gamma_max": gamma,
            "dicm_reliable_contrast_min_reliability": .5,
            "dicm_reliable_contrast_warmup_steps": 300,
            "dicm_reliable_contrast_ramp_steps": 200,
            "dicm_rcl_diagnostics": True}


def command(dataset, group, target, config):
    return [sys.executable, "-B", "-u", "train_all.py",
            f"reliability_{dataset}_{group}_env{target}", str(config),
            "--algorithm", "DICM", "--dataset", dataset, "--test_envs", str(target),
            "--steps", "5001", "--checkpoint_freq", "100", "--seed", "0", "--trial_seed", "0",
            "--deterministic", "--holdout_fraction", "0.2", "--label_ratio", "1.0",
            "--model", "resnet50", "--data_dir", str(DATA), "--output_root", str(ROOT / "output")]


def preflight():
    verify_source()
    for d in DOMAINS:
        for group, gamma in GAMMAS.items():
            config = ROOT / "configs" / f"{d}_{group}.yaml"
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(yaml.safe_dump(make_config(d, gamma), sort_keys=True), encoding="utf8")
            cmd = [*command(d, group, 0, config), "--show"]
            out = subprocess.check_output(cmd, cwd=PROJECT, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            key = "Requested steps=5001; max_updates=5001; checkpoint_freq=100\n"
            assert key in out
            effective = yaml.safe_load(re.sub(r"\x1b\[[0-9;]*m", "", out).split(key, 1)[1])
            for k, expected in make_config(d, gamma).items():
                actual = effective[k]
                if isinstance(expected, float):
                    actual = float(actual)
                if actual != expected:
                    raise ValueError(f"{d} {group}: {k}={actual}, expected {expected}")
            (ROOT / "configs" / f"{d}_{group}_effective.txt").write_text(out, encoding="utf8")
        say(f"{d}: three historical-configuration variants preflighted")
    (ROOT / "protocol.txt").write_text(PROTOCOL + "\nB0 gate: four converged folds, mean at least "
        "historical minus 1 percentage point and each domain at least historical minus 3 percentage points.\n",
        encoding="utf8")


def gpu_ready(gpu):
    while True:
        free = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                                           "--format=csv,noheader,nounits"], text=True).strip())
        meminfo = Path("/proc/meminfo").read_text()
        mem_avail = int(re.search(r"MemAvailable:\s+(\d+)", meminfo)[1]) / 1024**2
        disk = shutil.disk_usage(ROOT).free / 1024**3
        if free > 30720 and mem_avail > 12 and disk > 5:
            return free
        say(f"GPU{gpu}: free {free} MiB; system available {mem_avail:.1f} GiB; disk {disk:.1f} GiB")
        time.sleep(60)


def publish_summary(dataset, group, results, gate):
    import tempfile
    payload = {"dataset": dataset, "group": group, "gamma": GAMMAS[group],
               "code_commit": SOURCE, "seed": 0, "trial_seed": 0, "protocol": PROTOCOL,
               "folds": results, "iid_mean": statistics.mean(x["iid"] for x in results),
               "swad_mean": (statistics.mean(x["swad"] for x in results)
                             if all(x["converged"] for x in results) else None), "B0_gate": gate}
    text = archive.safe(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    ref = "refs/heads/experiments/dicm"
    relative = Path("reliability_paired_20261118") / dataset / (group + ".json")
    for attempt in range(5):
        try:
            with tempfile.TemporaryDirectory(prefix="dicm-reliability-archive-") as temp:
                work = Path(temp)
                archive.git(work, "init", "-q")
                archive.git(work, "fetch", "--depth=1", archive.REMOTE, ref)
                archive.git(work, "checkout", "--detach", "FETCH_HEAD")
                dest = work / relative
                if dest.exists():
                    if dest.read_text() != text:
                        raise ValueError("Archive summary already exists with different contents")
                    return archive.git(work, "rev-parse", "HEAD").decode().strip()
                for item in results:
                    if not (work / "runs" / item["run_id"] / "evaluation.json").is_file():
                        raise ValueError("Missing individual fold archive")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(text, encoding="utf8")
                archive.git(work, "add", "--", str(relative))
                archive.git(work, "commit", "-qm", f"Archive DICM reliability {dataset} {group}")
                archive.git(work, "push", archive.REMOTE, "HEAD:" + ref)
                archive.git(work, "fetch", "--depth=1", archive.REMOTE, ref)
                assert archive.git(work, "show", "FETCH_HEAD:" + str(relative)).decode() == text
                return archive.git(work, "rev-parse", "FETCH_HEAD").decode().strip()
        except (OSError, subprocess.SubprocessError, AssertionError) as exc:
            say(f"Archive retry {attempt + 1}: {type(exc).__name__}")
            time.sleep(20)
    raise RuntimeError("Could not synchronize group summary")


def get_fold(dataset, group, target, gpu):
    name = f"reliability_{dataset}_{group}_env{target}"
    outdir = ROOT / "output" / dataset
    candidates = list(outdir.glob("*_" + name))
    completed = [p for p in candidates if (p / ".github-archive" / "evaluation.json").is_file()]
    if len(completed) > 1:
        raise ValueError("Duplicate completed fold")
    if not completed:
        if candidates:
            raise RuntimeError(f"Incomplete previous run requires inspection: {name}")
        free = gpu_ready(gpu)
        config = ROOT / "configs" / f"{dataset}_{group}.yaml"
        cmd = command(dataset, group, target, config)
        console = ROOT / "console" / f"{dataset}_{group}_env{target}.log"
        console.parent.mkdir(exist_ok=True)
        gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid",
                                             "--format=csv,noheader"], text=True).strip()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        say(f"{name}: starting on GPU{gpu} with {free} MiB free")
        with console.open("x", encoding="utf8", buffering=1) as stream:
            proc = subprocess.run(cmd, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f"{name} exited {proc.returncode}; inspect {console}")
        completed = [p for p in outdir.glob("*_" + name) if (p / ".github-archive" / "evaluation.json").is_file()]
    if len(completed) != 1:
        raise ValueError("Missing completed fold archive")
    run = completed[0]
    box = run / ".github-archive"
    manifest = json.loads((box / "manifest.json").read_text())
    fold, = json.loads((box / "evaluation.json").read_text())
    h = fold["hparams"]
    for k, expected in make_config(dataset, GAMMAS[group]).items():
        if h.get(k) != expected:
            raise ValueError(f"{name}: effective hparam {k} differs")
    if manifest["code"]["commit"] != SOURCE or fold["target"] != [DOMAINS[dataset][target]]:
        raise ValueError("Source version or held-out target does not match")
    swad = fold["swad_evaluation"]
    if not swad or any(k not in swad for k in ("converged", "dead_valley", "stop_update", "metrics")):
        raise ValueError("SWAD state is missing")
    if not math.isclose(swad["metrics"]["test_in"], fold["selection"]["SWAD"]):
        raise ValueError("SWAD output mismatch")
    diag_keys = ["rcl_pair_count", "rcl_weight_mean", "rcl_weight_std", "rcl_weight_q10",
                 "rcl_weight_q50", "rcl_weight_q90", "rcl_weight_below_099", "rcl_weight_below_095",
                 "rcl_weight_below_090", "rcl_plain_loss", "rcl_loss_difference",
                 "rcl_backbone_gradient_norm", "plain_backbone_gradient_norm",
                 "rcl_gradient_difference_norm", "rcl_plain_gradient_cosine"]
    diagnostics = []
    for row in fold["checkpoints"]:
        if any(key not in row for key in diag_keys):
            raise ValueError("Missing reliability diagnostics")
        diagnostics.append({"step": row["step"], "gamma": row["reliable_contrast_gamma"]
                            if "reliable_contrast_gamma" in row else 0.0,
                            **{k: row[k] for k in diag_keys}})
    result = {"target": target, "domain": DOMAINS[dataset][target], "iid": fold["selection"]["iid"],
              "swad": fold["selection"]["SWAD"], "best_source_validation": fold["iid_checkpoint"]["train_out"],
              "converged": swad["converged"], "fallback": not swad["converged"],
              "dead_valley": swad["dead_valley"], "converge_step": swad["converge_step"],
              "stop_update": swad["stop_update"], "window": [swad["start"], swad["end"]],
              "run_id": manifest["run_id"], "diagnostics": diagnostics}
    archive.publish(box)
    return result


def run(dataset, gpu):
    verify_source()
    with (ROOT / f"{dataset}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        all_groups = {}
        for group in GAMMAS:
            folds = [get_fold(dataset, group, target, gpu) for target in range(4)]
            all_groups[group] = folds
            gate = None
            if group == "B0":
                historical = HISTORICAL_SWAD[dataset]
                differences = [x["swad"] - h for x, h in zip(folds, historical)]
                passed = all(x["converged"] for x in folds) and statistics.mean(differences) >= -.01 and min(differences) >= -.03
                gate = {"passed": passed, "mean_delta": statistics.mean(differences),
                        "domain_deltas": differences, "mean_tolerance": .01, "domain_tolerance": .03}
            commit = publish_summary(dataset, group, folds, gate)
            state = {"dataset": dataset, "last_group": group, "results": all_groups, "B0_gate": gate}
            (ROOT / f"{dataset}_status.json").write_text(json.dumps(state, indent=2) + "\n")
            say(f"{dataset} {group} completed: IID {statistics.mean(x['iid'] for x in folds):.3%}; "
                f"SWAD {statistics.mean(x['swad'] for x in folds):.3%}; archive {commit}")
            if gate is not None and not gate["passed"]:
                say(f"{dataset} B0 gate failed. Stop and inspect historical code, seed protocol and settings.")
                return
        say(f"{dataset} B0/B1/B2 finished. Stop before any additional search.")


if __name__ == "__main__":
    if sys.argv[1:] == ["preflight"]:
        preflight()
    elif len(sys.argv) == 3 and (sys.argv[1], sys.argv[2]) in (("PACS", "0"), ("VLCS", "1")):
        run(sys.argv[1], int(sys.argv[2]))
    else:
        raise SystemExit("Usage: run.py preflight | run.py PACS 0 | run.py VLCS 1")
