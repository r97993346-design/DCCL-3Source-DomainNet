"""Independent PACS study using the existing CIPT fold runner."""
import argparse
import fcntl
import itertools
import os
import random
import signal
import subprocess
import sys
from pathlib import Path

import optuna
from optuna.trial import TrialState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cipt_clip_supcon_optuna"))
import protocol as p
import run as runner

NAME = "pacs_fixed6_tpe_prompt_rd3_condcov3_20261127"
ROOT = p.BASE / "outputs" / NAME
SPACE = {
    "condcov_weight": (0.5, 1.0, 2.0, 3.0),
    "rd_weight": (0.5, 1.0, 2.0, 3.0),
    "contrastive_weight": (0.0, 0.001, 0.025, 0.05),
    "contrastive_mix": (0.0, 0.25, 0.5, 0.75),
    "prompt_length": (4, 8, 16),
}
FIXED = [(0.5,0.5,0.001,0.,4), (0.5,0.5,0.,0.,4), (0.5,2.,0.001,0.,4),
         (0.5,2.,0.001,0.75,4), (2.,2.,0.001,0.75,4), (3.,3.,0.025,0.25,8)]
SEEDS = [280998037, 1975502969, 683537036, 990940125]
FIXED = [dict(zip(SPACE, row)) for row in FIXED]
PAUSE = ROOT / "pause.request"
FIXED_HPARAMS = p.fixed_hparams()


def effective(params):
    values = dict(params)
    if values["contrastive_weight"] == 0:
        values["contrastive_mix"] = 0.
    return tuple(values[key] for key in SPACE)


CANDIDATES = sorted({effective(dict(zip(SPACE, row))) for row in itertools.product(*SPACE.values())})


def training_params(values):
    return dict(zip(SPACE, effective(values)), initial_lr=2e-5)


def hparams(values):
    values = training_params(values)
    return dict(FIXED_HPARAMS, lr=2e-5,
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
runner.GPU_FOR_DATASET = {"PACS": int(os.environ.get("CIPT_GPU_PACS", "0"))}
runner.ACTIVE_GPU = dict.fromkeys(runner.GPU_FOR_DATASET.values(), 0)


def immutable(path, value):
    if path.exists():
        p.require(p.read_json(path) == value, f"persisted configuration differs: {path}")
    else:
        p.write_json(path, value)


def prepare():
    p.require(len(CANDIDATES) == 624 and len({effective(row) for row in FIXED}) == 6, "bad search space")
    p.require([p.seed_for("PACS", i) for i in range(4)] == SEEDS, "fold seed mismatch")
    p.require(p.CLIP.is_file() and (p.DATA / "PACS").is_dir(), "weights or PACS missing")
    ROOT.mkdir(parents=True, exist_ok=True)
    immutable(ROOT / "fixed_trials.json", FIXED)
    immutable(ROOT / "search_space.json", {k: list(v) for k, v in SPACE.items()})
    immutable(ROOT / "protocol.json", {
        "dataset": "PACS", "space_size": 624, "fixed_args": p.FIXED_ARGS,
        "fixed_hparams": FIXED_HPARAMS, "initial_lr": 2e-5, "scheduler": p.LR_POLICY,
        "seeds": SEEDS, "target_iid": .975, "objective": "mean four-fold source validation",
        "sampler": {"seed": 42, "n_startup_trials": 6, "n_ei_candidates": 24, "multivariate": False},
        "pruner": "NopPruner", "n_jobs": 1,
        "reference_t0": {"source_val": .98690, "iid": .96987, "test_out": .96934},
    })
    immutable(ROOT / "snapshot.json", {"commit": p.git("rev-parse", "HEAD"),
              "data": str(p.DATA), "weights": str(p.CLIP), "python": sys.version.split()[0],
              "optuna": optuna.__version__})


def ready():
    p.require(runner.available_ram_kib() > runner.MIN_RAM_KIB, "available RAM must exceed 10 GiB")
    gpu = runner.GPU_FOR_DATASET["PACS"]
    uuid = runner.gpu_uuids()[gpu]
    p.require(not p.git("status", "--porcelain", "--", "domainbed", "train_all.py", "scripts"), "training tree not clean")
    env = runner.environment_for(uuid)
    subprocess.run([sys.executable, "-B", "-c",
                    "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"],
                   env=env, check=True)
    return gpu, env


def tables(study):
    rows, folds = [], []
    for trial in study.get_trials(deepcopy=False):
        if not trial.params or not set(SPACE) <= trial.params.keys():
            continue
        params = training_params(trial.params)
        path = ROOT / "PACS" / "trials" / f"trial_{trial.number:03d}"
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
    hit = next((row for row in complete if row["iid"] >= .975), None)
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
    prepare()
    with (ROOT / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu, env = ready()
        study = optuna.create_study(study_name=NAME, direction="maximize", load_if_exists=True,
                storage=f"sqlite:///{ROOT / 'optuna.db'}",
                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=6, n_ei_candidates=24, multivariate=False),
                pruner=optuna.pruners.NopPruner())
        if not study.trials:
            for i, params in enumerate(FIXED):
                study.enqueue_trial(params, user_attrs={"origin": f"fixed_T{i}"})
        signal.signal(signal.SIGTERM, lambda *_: PAUSE.touch())
        while not PAUSE.exists():
            if tables(study):
                reason = "target_reached"
                break
            item = candidate(study)
            if item is None:
                reason = "space_exhausted"
                break
            trial, params = item
            result = runner.finish_trial(ROOT / "PACS", "PACS", study, trial.number, params, gpu, env, lock.fileno())
            trial.set_user_attr("result", result)
            if "error" in result:
                study.tell(trial, state=TrialState.FAIL)
            else:
                study.tell(trial, result["source_val"])
            tables(study)
            (ROOT / "sync.request").touch()
        else:
            reason = "paused"
        p.write_json(ROOT / "status.json", {"state": reason})
        (ROOT / "sync.request").touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "ready", "run", "pause", "resume"))
    action = parser.parse_args().action
    os.chdir(p.CODE)
    if action == "pause":
        PAUSE.touch()
    elif action == "prepare":
        prepare()
    elif action == "ready":
        ready()
    else:
        if action == "resume":
            with (ROOT / "controller.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                PAUSE.unlink(missing_ok=True)
        search()


if __name__ == "__main__":
    main()
