"""Independent PACS study using the existing CIPT fold runner."""
import argparse
from contextlib import ExitStack
import fcntl
import itertools
from importlib.util import module_from_spec, spec_from_file_location
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna
from optuna.trial import TrialState

legacy = Path(__file__).resolve().parents[1] / "cipt_clip_supcon_optuna"
sys.path.insert(0, str(legacy))
import protocol as p

runner_spec = spec_from_file_location("cip_t_fold_runner", legacy / "run.py")
runner = module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)

DATASET = "TerraIncognita"
INITIAL_LR = 2e-5
TARGET_IID = .58
SPACE_SIZE = 816
NAME = "terra_fixed6_tpe_lr2e5_prompt_rd3_condcov3_20261127"
ROOT = p.BASE / "outputs" / NAME
SPACE = {
    "condcov_weight": (0.5, 1.0, 2.0, 3.0),
    "rd_weight": (0.5, 1.0, 2.0, 3.0),
    "contrastive_weight": (0.0, 0.001, 0.025, 0.05, 0.1),
    "contrastive_mix": (0.0, 0.25, 0.5, 0.75),
    "prompt_length": (4, 8, 16),
}
FIXED = [(.5,1.,0.,0.,16), (.5,1.,0.,0.,4), (2.,2.,0.,0.,4),
         (2.,2.,.025,0.,4), (2.,2.,.025,.75,4), (3.,3.,.001,.25,8)]
DOMAINS = ("L100", "L38", "L43", "L46")
SEEDS = [p.seed_for(DATASET, fold) for fold in range(4)]
p.DATASETS[DATASET] = (DOMAINS, TARGET_IID, "terra_incognita_fixed")
p.PROTOCOL_ARGS["cipt_cache_eval_visual"] = True
legacy_command = p.command


def command(*args, **kwargs):
    return [*legacy_command(*args, **kwargs), "--cipt_cache_eval_visual"]


runner.command = command
FIXED = [dict(zip(SPACE, row)) for row in FIXED]
PAUSE = ROOT / "pause.request"
STOP = ROOT / "stop.request"
AUTHORIZATION = ROOT / "start.authorized"
FIXED_HPARAMS = p.fixed_hparams()


def effective(params):
    values = dict(params)
    if values["contrastive_weight"] == 0:
        values["contrastive_mix"] = 0.
    return tuple(values[key] for key in SPACE)


CANDIDATES = sorted({effective(dict(zip(SPACE, row))) for row in itertools.product(*SPACE.values())})


def training_params(values):
    return dict(zip(SPACE, effective(values)), initial_lr=INITIAL_LR)


def hparams(values):
    values = training_params(values)
    return dict(FIXED_HPARAMS, lr=INITIAL_LR,
                cipt_condcov_weight=values["condcov_weight"],
                cipt_env_loss_weight=values["rd_weight"],
                cipt_causal_contrastive_weight=values["contrastive_weight"],
                cipt_contrastive_weight=values["contrastive_weight"],
                cipt_clip_teacher_mix=values["contrastive_mix"],
                cipt_prompt_length=values["prompt_length"])


p.hparams_for = hparams
runner.hparams_for = hparams
runner.OUTPUT = ROOT
runner.PAUSE = PAUSE
GPU_REQUEST = os.environ.get("CIPT_GPU_TERRA")
runner.GPU_FOR_DATASET = {DATASET: int(GPU_REQUEST or "0")}
runner.ACTIVE_GPU = dict.fromkeys(runner.GPU_FOR_DATASET.values(), 0)


def immutable(path, value):
    if path.exists():
        p.require(p.read_json(path) == value, f"persisted configuration differs: {path}")
    else:
        p.write_json(path, value)


def prepare():
    p.require(len(CANDIDATES) == SPACE_SIZE and len({effective(row) for row in FIXED}) == 6, "bad search space")
    p.require([p.seed_for(DATASET, i) for i in range(4)] == SEEDS, "fold seed mismatch")
    p.require(p.CLIP.is_file() and (p.DATA / p.DATASETS[DATASET][2]).is_dir(), "weights or PACS missing")
    ROOT.mkdir(parents=True, exist_ok=True)
    immutable(ROOT / "fixed_trials.json", FIXED)
    immutable(ROOT / "search_space.json", {k: list(v) for k, v in SPACE.items()})
    immutable(ROOT / "protocol.json", {
        "dataset": DATASET, "space_size": SPACE_SIZE, "fixed_args": p.FIXED_ARGS,
        "fixed_hparams": FIXED_HPARAMS, "initial_lr": INITIAL_LR, "scheduler": p.LR_POLICY,
        "seeds": SEEDS, "domains": list(DOMAINS), "evaluation_cache": True, "target_iid": TARGET_IID, "objective": "mean four-fold source validation",
        "sampler": {"seed": 42, "n_startup_trials": 6, "n_ei_candidates": 24, "multivariate": False},
        "pruner": "NopPruner", "n_jobs": 1,
        "auto_start": False, "requires_explicit_start_authorization": True,
    })
    immutable(ROOT / "snapshot.json", {"commit": p.git("rev-parse", "HEAD"),
              "data": str(p.DATA / p.DATASETS[DATASET][2]), "weights": str(p.CLIP), "python": sys.version.split()[0],
              "optuna": optuna.__version__})


def require_authorization():
    p.require(AUTHORIZATION.is_file(), "Terra is paused pending explicit start authorization")


def hardware():
    require_authorization()
    p.require(runner.available_ram_kib() > runner.MIN_RAM_KIB, "available RAM must exceed 10 GiB")
    p.require(not p.git("status", "--porcelain", "--", "domainbed", "train_all.py", "scripts"), "training tree not clean")
    listing = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True, timeout=15)
    devices = dict((int(index.strip()), uuid.strip()) for index, uuid in
                   (line.split(",") for line in listing.splitlines() if line.strip()))
    p.require(devices and (GPU_REQUEST is None or int(GPU_REQUEST) in devices), "requested GPU unavailable")
    subprocess.run([sys.executable, "-B", "-c", "import torch; assert torch.cuda.is_available()"], check=True)
    return devices


def pacs_gpu():
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (proc / "cmdline").read_bytes().split(b"\0")
            if not any(arg.endswith(b"scripts/pacs_fixed6/run.py") for arg in cmd):
                continue
            if b"run" not in cmd and b"resume" not in cmd:
                continue
            env = dict(item.split(b"=", 1) for item in
                       (proc / "environ").read_bytes().split(b"\0") if b"=" in item)
            return int(env.get(b"CIPT_GPU_PACS", b"0"))
        except (OSError, ValueError):
            continue
    return None


def reserve_gpu(devices, stack):
    candidates = [int(GPU_REQUEST)] if GPU_REQUEST is not None else sorted(devices)
    while not PAUSE.exists() and not STOP.exists():
        occupied = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True, timeout=15)
        for gpu in candidates:
            if gpu == pacs_gpu() or devices[gpu] in occupied or runner.available_ram_kib() <= runner.MIN_RAM_KIB:
                continue
            reservation = (p.BASE / f".fixed6-gpu-{gpu}.lock").open("a")
            try:
                fcntl.flock(reservation, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                reservation.close()
                continue
            stack.enter_context(reservation)
            runner.GPU_FOR_DATASET = {DATASET: gpu}
            runner.ACTIVE_GPU = {gpu: 0}
            return gpu, runner.environment_for(devices[gpu])
        p.write_json(ROOT / "status.json", {"state": "waiting_for_gpu"})
        time.sleep(15)
    return None


def tables(study):
    rows, folds = [], []
    for trial in study.get_trials(deepcopy=False):
        if not trial.params or not set(SPACE) <= trial.params.keys():
            continue
        params = training_params(trial.params)
        path = ROOT / DATASET / "trials" / f"trial_{trial.number:03d}"
        result = trial.user_attrs.get("result", {})
        rows.append(dict(trial=trial.number, state=trial.state.name, **params,
                         origin=trial.user_attrs.get("origin"), **{k: result.get(k) for k in
                         ("source_val", "iid", "test_in", "test_out", "elapsed_seconds", "error")}))
        for marker in sorted(path.glob("fold_*.json")):
            folds.append(dict(trial=trial.number, **p.read_json(marker)))
    columns = ["trial", "state", *SPACE, "initial_lr", "origin", "source_val", "iid", "test_in", "test_out", "elapsed_seconds", "error"]
    p.write_csv(ROOT / "trials_summary.csv", rows, columns)
    p.write_csv(ROOT / "folds_summary.csv", folds, list(folds[0]) if folds else ["trial", "fold", "domain", "source_val"])
    complete = [row for row in rows if row["state"] == "COMPLETE"]
    if complete:
        for key in ("source_val", "iid", "test_out"):
            p.write_json(ROOT / f"best_{key}.json", max(complete, key=lambda row: row[key]))
    hit = next((row for row in complete if row["iid"] >= TARGET_IID), None)
    if hit and sum(row["trial"] < 6 for row in complete) == 6:
        p.write_json(ROOT / "target_reached.json", hit)
        return True
    return False


def refresh_fold(dataset, study, root, state, **kwargs):
    tables(study)
    p.write_json(ROOT / "status.json", {"state": state, **kwargs})


runner.refresh = refresh_fold


def candidate(study):
    trials = study.get_trials(deepcopy=False)
    active = [t for t in trials if t.state == TrialState.RUNNING]
    p.require(len(active) <= 1, "multiple running trials")
    fixed_done = all(t.state == TrialState.COMPLETE for t in trials[:6])
    if not fixed_done and any(t.state == TrialState.FAIL for t in trials[:6]):
        raise RuntimeError("fixed trial failed after fold retries; inspect attempts before continuing")
    blocked = {effective(t.params) for t in trials
               if set(SPACE) <= t.params.keys()
               and (t.state == TrialState.COMPLETE or
                    (t.state == TrialState.FAIL and t.user_attrs.get("origin") != "duplicate"))}
    for attempt in range(25):
        if active:
            trial = optuna.Trial(study, active.pop()._trial_id)
        else:
            if fixed_done and len(blocked) == len(CANDIDATES):
                return None
            if attempt == 24:
                choice = random.Random(42 + len(trials)).choice([row for row in CANDIDATES if row not in blocked])
                study.enqueue_trial(dict(zip(SPACE, choice)), user_attrs={"origin": "fallback"})
            trial = study.ask()
        raw = {key: trial.suggest_categorical(key, choices) for key, choices in SPACE.items()}
        if effective(raw) not in blocked:
            trial.set_user_attr("origin", trial.user_attrs.get("origin", "tpe"))
            return trial, training_params(raw)
        trial.set_user_attr("origin", "duplicate")
        study.tell(trial, state=TrialState.FAIL)
    raise RuntimeError("fallback did not produce an available candidate")


def search():
    require_authorization()
    prepare()
    with (ROOT / "controller.lock").open("a") as lock, ExitStack() as stack:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assignment = reserve_gpu(hardware(), stack)
        if assignment is None:
            p.write_json(ROOT / "status.json", {"state": "stopped" if STOP.exists() else "paused"})
            (ROOT / "sync.request").touch()
            return
        gpu, env = assignment
        study = optuna.create_study(study_name=NAME, direction="maximize", load_if_exists=True,
                storage=f"sqlite:///{ROOT / 'optuna.db'}",
                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=6, n_ei_candidates=24, multivariate=False),
                pruner=optuna.pruners.NopPruner())
        if not study.trials:
            for i, params in enumerate(FIXED):
                study.enqueue_trial(params, user_attrs={"origin": f"fixed_T{i}"})
        signal.signal(signal.SIGTERM, lambda *_: STOP.touch())
        while not PAUSE.exists() and not STOP.exists():
            if tables(study):
                reason = "target_reached"
                break
            item = candidate(study)
            if item is None:
                reason = "space_exhausted"
                break
            trial, params = item
            result = runner.finish_trial(ROOT / DATASET, DATASET, study, trial.number, params, gpu, env, lock.fileno())
            trial.set_user_attr("result", result)
            if "error" in result:
                study.tell(trial, state=TrialState.FAIL)
            else:
                study.tell(trial, result["source_val"])
            tables(study)
            (ROOT / "sync.request").touch()
        else:
            reason = "stopped" if STOP.exists() else "paused"
        p.write_json(ROOT / "status.json", {"state": reason})
        (ROOT / "sync.request").touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "ready", "run", "pause", "resume", "stop", "authorize-start"))
    action = parser.parse_args().action
    os.chdir(p.CODE)
    if action in ("pause", "stop"):
        ROOT.mkdir(parents=True, exist_ok=True)
        (PAUSE if action == "pause" else STOP).touch()
    elif action == "prepare":
        prepare()
    elif action == "authorize-start":
        prepare()
        with (ROOT / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not AUTHORIZATION.exists():
                AUTHORIZATION.touch()
                PAUSE.unlink(missing_ok=True)
    elif action == "ready":
        print(hardware())
    else:
        require_authorization()
        if action == "resume":
            with (ROOT / "controller.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                PAUSE.unlink(missing_ok=True)
                STOP.unlink(missing_ok=True)
        search()


if __name__ == "__main__":
    main()
