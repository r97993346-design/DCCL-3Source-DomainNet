"""Continuous end-to-end DICM HPO using held-out-domain SWAD (oracle-guided)."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import optuna
from optuna.trial import Trial, TrialState
import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import experiment_archive

ROOT = Path("/home/hooasia/lgg/projects/dicm-oracle-swad-optuna-20261117")
DATA = Path("/home/hooasia/lgg/data/repro_dccl_data")
REMOTE = experiment_archive.REMOTE
BRANCH = "refs/heads/experiments/dicm"
DOMAINS = {
    "PACS": ("art_painting", "cartoon", "photo", "sketch"),
    "VLCS": ("CALTECH", "LABELME", "PASCAL", "SUN"),
}
PROTOCOL = "oracle-guided HPO"
STEPS = 5001
CHECKPOINT_FREQ = 100
FIXED = {
    "dicm_ablation": "A4", "dicm_residual_projection": True,
    "dicm_rank": 8, "dicm_beta": 0.75,
    "dicm_aug_weight": 0.25, "dicm_dom_weight": 0.75,
    "l_layer": 0.0, "l_d": 0.0, "dicm_gt_weight": 0.0,
    "batch_size": 32, "label_ratio": 1.0,
    "swad": "LossValley",
    "swad_kwargs": {"n_converge": 3, "n_tolerance": 6, "tolerance_ratio": 0.3},
    "dicm_reliable_contrast_min_reliability": 0.5,
    "dicm_reliable_contrast_warmup_steps": 300,
    "dicm_reliable_contrast_ramp_steps": 200,
    "TN": False, "re_w": False, "sample_d": False,
}
RANGES = {
    "lr": [1e-5, 1e-4, "log"], "dicm_basis_lr": [1e-5, 3e-4, "log"],
    "l": [0.1, 2.0, "log"], "dicm_isr_weight": [0.005, 0.1, "log"],
    "dicm_reliable_contrast_gamma_max": [0.1, 1.0, "linear"],
    "weight_decay": [1e-6, 1e-3, "log"], "sgd_momentum": [0.8, 0.99, "linear"],
    "optimizer": ["adam", "adamw", "sgd"], "lr_schedule": ["fixed", "cosine"],
}


def say(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S %Z"), message, flush=True)


def code(*args):
    return subprocess.check_output(["git", "-C", str(PROJECT), *args], text=True).strip()


def validate_code(expected):
    if code("rev-parse", "HEAD") != expected or code("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Source commit or tracked working tree changed")


def sample(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-4, log=True),
        "dicm_basis_lr": trial.suggest_float("dicm_basis_lr", 1e-5, 3e-4, log=True),
        "l": trial.suggest_float("l", 0.1, 2.0, log=True),
        "dicm_isr_weight": trial.suggest_float("dicm_isr_weight", 0.005, 0.1, log=True),
        "dicm_reliable_contrast_gamma_max": trial.suggest_float(
            "dicm_reliable_contrast_gamma_max", 0.1, 1.0),
        "lr_schedule": trial.suggest_categorical("lr_schedule", ["fixed", "cosine"]),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }
    if params["optimizer"] == "sgd":
        params["sgd_momentum"] = trial.suggest_float("sgd_momentum", 0.8, 0.99)
    return params


def config_for(params):
    return {**FIXED, **params}


def fingerprint(dataset, params):
    text = json.dumps({"dataset": dataset, "params": params}, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def fold_result(dataset, target, fold, manifest, expected_code):
    if manifest["dataset"] != dataset or manifest["code"]["commit"] != expected_code:
        raise ValueError("Incorrect dataset or source commit in fold archive")
    if manifest.get("evaluation_protocol") != PROTOCOL:
        raise ValueError("Fold is not labelled oracle-guided HPO")
    args = fold["args"]
    if args.get("stage") is not None or args.get("stage_parent") is not None:
        raise ValueError("A fold attempted stagewise training")
    if fold["target"] != [DOMAINS[dataset][target]]:
        raise ValueError("Incorrect target in fold archive")
    h = fold["hparams"]
    if (h["dicm_ablation"] != "A4" or not h["dicm_residual_projection"]
            or float(h["l"]) <= 0 or float(h["l_layer"]) != 0 or float(h["l_d"]) != 0
            or float(h["dicm_reliable_contrast_gamma_max"]) <= 0
            or h["swad"] != "LossValley"
            or h["swad_kwargs"] != FIXED["swad_kwargs"]):
        raise ValueError("Fold model or SWAD parameters differ from the requested structure")
    swad = fold.get("swad_evaluation")
    if not swad or any(key not in swad for key in (
            "converged", "dead_valley", "converge_step", "first_dead_update",
            "stop_update", "start", "end", "count", "metrics")):
        raise ValueError("Missing SWAD convergence or stop metadata")
    iid = float(fold["selection"]["iid"])
    target_score = float(swad["metrics"]["test_in"])
    source_score = float(fold["iid_checkpoint"]["train_out"])
    if target_score != float(fold["selection"]["SWAD"]):
        raise ValueError("SWAD selection and final-model test_in disagree")
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (iid, target_score, source_score)):
        raise ValueError("Non-finite or out-of-range metrics")
    if swad["converged"] != (swad["converge_step"] is not None):
        raise ValueError("Inconsistent convergence metadata")
    if swad["dead_valley"] != (swad["first_dead_update"] is not None):
        raise ValueError("Inconsistent valley-stop metadata")
    return {
        "target": target, "domain": DOMAINS[dataset][target],
        "iid": iid, "swad": target_score, "best_source_validation": source_score,
        "swad_source_validation": float(swad["metrics"]["train_out"]),
        "resolved_hparams": h,
        "source_best_step": fold["iid_checkpoint"]["step"],
        "converged": swad["converged"], "fallback": not swad["converged"],
        "dead_valley": swad["dead_valley"], "converge_step": swad["converge_step"],
        "first_dead_update": swad["first_dead_update"], "stop_update": swad["stop_update"],
        "swad_start": swad["start"], "swad_end": swad["end"], "swad_count": swad["count"],
        "archive_run_id": manifest["run_id"],
    }


def objective_value(folds):
    if len(folds) != 4 or {r["target"] for r in folds} != set(range(4)):
        raise ValueError("Trial does not contain exactly four target folds")
    invalid = [r["domain"] for r in folds if r["fallback"] or not r["converged"]]
    if invalid:
        return None, "SWAD did not converge in " + ", ".join(invalid)
    return sum(r["swad"] for r in folds) / 4, None


def wait_for_gpu(gpu):
    while True:
        free_disk = shutil.disk_usage(ROOT).free
        if free_disk < 2 * 1024 ** 3:
            say(f"Waiting for disk space on {ROOT}")
            time.sleep(60)
            continue
        output = subprocess.check_output([
            "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
            "--format=csv,noheader,nounits"], text=True)
        free = int(output.strip())
        if free > 30720:
            return free
        say(f"GPU {gpu} free memory {free} MiB; waiting for >30720 MiB")
        time.sleep(60)


def publish_trial(dataset, trial, payload):
    contents = experiment_archive.safe(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                  indent=2, allow_nan=False) + "\n")
    relative = Path("oracle_hpo") / "dicm_swad_20261117" / dataset / f"trial_{trial.number:06d}.json"
    for _ in range(3):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="dicm-oracle-archive-") as temporary:
            work = Path(temporary)
            experiment_archive.git(work, "init", "-q")
            experiment_archive.git(work, "fetch", "--depth=1", REMOTE, BRANCH)
            experiment_archive.git(work, "checkout", "--detach", "FETCH_HEAD")
            path = work / relative
            if path.exists():
                if path.read_text(encoding="utf8") != contents:
                    raise ValueError("Archived trial report differs")
                return experiment_archive.git(work, "rev-parse", "HEAD").decode().strip()
            for fold in payload["folds"]:
                if not (work / "runs" / fold["archive_run_id"] / "evaluation.json").is_file():
                    raise ValueError("Fold archive is missing on GitHub")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf8")
            experiment_archive.git(work, "add", "--", str(relative))
            experiment_archive.git(work, "commit", "-qm",
                                   f"Archive oracle-guided DICM SWAD {dataset} trial {trial.number}")
            try:
                experiment_archive.git(work, "push", REMOTE, "HEAD:" + BRANCH)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
            experiment_archive.git(work, "fetch", "--depth=1", REMOTE, BRANCH)
            if experiment_archive.git(work, "show", "FETCH_HEAD:" + str(relative)).decode() == contents:
                return experiment_archive.git(work, "rev-parse", "FETCH_HEAD").decode().strip()
    raise RuntimeError("Could not verify GitHub trial report")


def get_fold(dataset, target, trial, params, commit, gpu, run_root):
    name = f"oracle_swad_{dataset}_trial{trial.number:06d}_env{target}"
    candidates = list((run_root / "output" / dataset).glob("*_" + name))
    complete = [p for p in candidates if (p / ".github-archive" / "evaluation.json").is_file()]
    if len(complete) > 1:
        raise ValueError("Multiple completed fold archives")
    if complete:
        run = complete[0]
    else:
        pid_path = run_root / f"env{target}.pid"
        if pid_path.exists():
            pid = int(pid_path.read_text())
            proc = Path("/proc") / str(pid)
            while proc.exists() and name in (proc / "cmdline").read_bytes().decode(errors="ignore"):
                say(f"{name}: existing training PID {pid} is still running")
                time.sleep(60)
            candidates = list((run_root / "output" / dataset).glob("*_" + name))
            complete = [p for p in candidates if (p / ".github-archive" / "evaluation.json").is_file()]
            if len(complete) > 1:
                raise ValueError("Multiple completed fold archives")
            if complete:
                return get_fold(dataset, target, trial, params, commit, gpu, run_root)
        for p in candidates:
            if (p / "results.jsonl").exists():
                rows = (p / "results.jsonl").read_text(encoding="utf8").splitlines()
                if rows and "=== Summary ===" in (p / "log.txt").read_text(encoding="utf8"):
                    raise RuntimeError(f"Training finished but archive is incomplete: {p}")
        free = wait_for_gpu(gpu)
        config = run_root / "config.yaml"
        command = [sys.executable, "-B", "-u", "train_all.py", name, str(config),
                   "--algorithm", "DICM", "--dataset", dataset, "--test_envs", str(target),
                   "--steps", str(STEPS), "--checkpoint_freq", str(CHECKPOINT_FREQ),
                   "--seed", "0", "--trial_seed", "0", "--deterministic",
                   "--label_ratio", "1.0", "--model", "resnet50",
                   "--data_dir", str(DATA), "--output_root", str(run_root / "output"),
                   "--oracle_guided_hpo"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
        uid = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid",
                                       "--format=csv,noheader"], text=True).strip()
        env["CUDA_VISIBLE_DEVICES"] = uid
        console = run_root / f"env{target}.log"
        with console.open("a", encoding="utf8", buffering=1) as log:
            log.write(f"GPU {gpu} {uid}; source {commit}; command {shlex.join(command)}\n")
            proc = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
            pid_path.write_text(str(proc.pid), encoding="utf8")
            say(f"{name}: started PID {proc.pid}, GPU free {free} MiB")
            exit_code = proc.wait()
        if exit_code:
            raise RuntimeError(f"{name} exited {exit_code}; see {console}")
        candidates = list((run_root / "output" / dataset).glob("*_" + name))
        complete = [p for p in candidates if (p / ".github-archive" / "evaluation.json").is_file()]
        if len(complete) != 1:
            raise RuntimeError(f"Fold output or GitHub archive missing for {name}")
        run = complete[0]
    outbox = run / ".github-archive"
    manifest = json.loads((outbox / "manifest.json").read_text(encoding="utf8"))
    fold, = json.loads((outbox / "evaluation.json").read_text(encoding="utf8"))
    for key, value in config_for(params).items():
        actual = fold["hparams"].get(key)
        if actual != value:
            raise ValueError(f"{name}: effective parameter {key}={actual!r}, expected {value!r}")
    result = fold_result(dataset, target, fold, manifest, commit)
    commit_archive = experiment_archive.publish(outbox)
    say(f"{name}: source={result['best_source_validation']:.4f}, "
        f"IID={result['iid']:.4f}, SWAD={result['swad']:.4f}, "
        f"fallback={result['fallback']}; archive={commit_archive}")
    return result


def loop(dataset, gpu):
    ROOT.mkdir(parents=True, exist_ok=True)
    dataset_root = ROOT / dataset
    dataset_root.mkdir(exist_ok=True)
    with (dataset_root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        commit = code("rev-parse", "HEAD")
        validate_code(commit)
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{dataset_root / 'oracle_swad.sqlite3'}",
            engine_kwargs={"connect_args": {"timeout": 60}})
        study = optuna.create_study(
            study_name=f"dicm_{dataset.lower()}_oracle_guided_swad_v1", storage=storage,
            direction="maximize", load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=int(time.time_ns() % (2 ** 32))))
        protocol = {"name": PROTOCOL, "objective": "mean held-out-domain SWAD test_in",
                    "commit": commit, "steps": STEPS, "checkpoint_freq": CHECKPOINT_FREQ,
                    "seed": 0, "trial_seed": 0, "fixed": FIXED, "ranges": RANGES}
        old = study.user_attrs.get("protocol")
        if old is not None and old != protocol:
            raise ValueError("Existing study uses a different protocol or code version")
        study.set_user_attr("protocol", protocol)
        say(f"{dataset} study started. {PROTOCOL}; objective is four-domain SWAD")
        while True:
            validate_code(commit)
            running = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
            if len(running) > 1:
                raise RuntimeError("Multiple unfinished trials in the same study")
            trial = Trial(study, running[0]._trial_id) if running else study.ask()
            params = sample(trial)
            digest = fingerprint(dataset, params)
            duplicate = next((t for t in study.get_trials(deepcopy=False)
                              if t.number != trial.number and
                              t.user_attrs.get("fingerprint") == digest and
                              t.state == TrialState.COMPLETE), None)
            if duplicate is not None:
                trial.set_user_attr("duplicate_of", duplicate.number)
                study.tell(trial, state=TrialState.PRUNED)
                continue
            trial.set_user_attr("fingerprint", digest)
            trial.set_user_attr("protocol", PROTOCOL)
            trial.set_user_attr("code_commit", commit)
            trial.set_user_attr("effective_config", config_for(params))
            run_root = dataset_root / f"trial_{trial.number:06d}"
            run_root.mkdir(exist_ok=True)
            config_file = run_root / "config.yaml"
            config_text = yaml.safe_dump(config_for(params), sort_keys=True)
            if config_file.exists() and config_file.read_text() != config_text:
                raise ValueError("Persisted trial configuration changed")
            config_file.write_text(config_text, encoding="utf8")
            folds = []
            for target in range(4):
                validate_code(commit)
                fold = get_fold(dataset, target, trial, params, commit, gpu, run_root)
                if folds and fold["resolved_hparams"] != folds[0]["resolved_hparams"]:
                    raise ValueError("The four folds used different resolved hyperparameters")
                folds.append(fold)
                trial.set_user_attr("folds", folds)
                trial.set_user_attr("resolved_hparams", fold["resolved_hparams"])
            value, reason = objective_value(folds)
            mean_iid = sum(row["iid"] for row in folds) / 4
            payload = {
                "protocol": PROTOCOL, "dataset": dataset, "trial": trial.number,
                "code_commit": commit, "objective": "mean held-out-domain SWAD test_in",
                "state": "invalid_fallback" if reason else "complete",
                "invalid_reason": reason, "iid_mean": mean_iid, "swad_mean": value,
                "folds": folds, "sampled_params": params, "effective_config": config_for(params),
                "seed": 0, "trial_seed": 0, "steps": STEPS, "checkpoint_freq": CHECKPOINT_FREQ,
            }
            trial.set_user_attr("iid_mean", mean_iid)
            trial.set_user_attr("swad_mean", value)
            trial.set_user_attr("invalid_reason", reason)
            archive_commit = publish_trial(dataset, trial, payload)
            trial.set_user_attr("archive_commit", archive_commit)
            if reason:
                trial.set_user_attr("invalid", True)
                study.tell(trial, state=TrialState.PRUNED)
                say(f"{dataset} trial {trial.number} INVALID: {reason}; archive {archive_commit}")
            else:
                study.tell(trial, value)
                say(f"{dataset} trial {trial.number} SWAD {value:.4%}; best {study.best_value:.4%}; "
                    f"oracle-guided HPO; archive {archive_commit}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in DOMAINS or sys.argv[2] not in ("0", "1"):
        raise SystemExit("Usage: python search.py PACS 0 | VLCS 1")
    loop(sys.argv[1], int(sys.argv[2]))
