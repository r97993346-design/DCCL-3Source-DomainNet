"""Stop launching old folds, let active folds exit, and record their results."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


BASE = Path(__file__).resolve().parents[5]
OLD_NAME = "condcov_optuna_joint_20261125"
OLD_CODE = BASE.with_name(OLD_NAME) / "code" / "DCCL" / "DCCL"
OLD_UNIT = "condcov-optuna-joint-20261125.service"
STATE = Path(__file__).resolve().parents[2] / "train_output" / f"{OLD_NAME}_low_contrast" / "transition.json"


def progress(state, **fields):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(state=state, previous_service=OLD_UNIT, **fields), indent=2) + "\n")
    temp.replace(STATE)
    print("contrast_space_transition", state, fields, flush=True)


def systemctl(*args):
    return subprocess.check_output(["systemctl", "--user", *args], text=True).strip()


def main_pid():
    return int(systemctl("show", OLD_UNIT, "-p", "MainPID", "--value"))


def process_state(stat):
    try:
        return stat.read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return None


def training_children(proc):
    children = set()
    for task in (proc / "task").iterdir():
        try:
            children.update(map(int, (task / "children").read_text().split()))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return sorted(pid for pid in children
                  if process_state(Path("/proc") / str(pid) / "stat") not in (None, "Z", "X"))


def drain():
    pid = main_pid()
    if pid:
        proc = Path("/proc") / str(pid)
        if (proc / "cwd").resolve(strict=True) != OLD_CODE:
            raise RuntimeError("previous service PID does not belong to the expected checkout")
        cmdline = (proc / "cmdline").read_bytes().split(b"\0")
        if b"scripts/condcov_optuna/run.py" not in cmdline or main_pid() != pid:
            raise RuntimeError("previous service PID changed or is not the Optuna scheduler")
        # Stop only the scheduler process. Its already-launched wrappers keep running.
        os.kill(pid, signal.SIGSTOP)
        while True:
            states = [process_state(task / "stat") for task in (proc / "task").iterdir()]
            if all(state in ("T", "t", "Z", "X", None) for state in states):
                break
            time.sleep(0.2)
        waiting = None
        while main_pid() == pid:
            workers = training_children(proc)
            if workers != waiting:
                progress("draining", scheduler_pid=pid, active_fold_wrapper_pids=workers)
                waiting = workers
            if not workers:
                break
            time.sleep(15)
        if main_pid() not in (0, pid):
            raise RuntimeError("previous Optuna service restarted during handoff")
    systemctl("stop", OLD_UNIT)
    if main_pid():
        raise RuntimeError("previous Optuna service still running")
    progress("reconciling")


def reconcile():
    sys.path.insert(0, str(OLD_CODE / "scripts" / "condcov_optuna"))
    import run as legacy

    for dataset in legacy.DATASETS:
        root = legacy.OUTPUT / dataset
        study, _ = legacy.study_for(dataset, root)
        pending = study.get_trials(deepcopy=False, states=(legacy.TrialState.RUNNING,))
        for trial in pending:
            if set(trial.params) != set(legacy.SPACE):
                continue
            directory = root / "trials" / f"trial_{trial.number:03d}"
            folds = []
            for fold in range(len(legacy.DATASETS[dataset][0])):
                for attempt in sorted((directory / f"env{fold}").glob("attempt_*"), reverse=True):
                    try:
                        saved = legacy.read_attempt(attempt, dataset, fold, trial.params)
                    except (KeyError, ValueError, OSError):
                        continue
                    folds.append(saved)
                    legacy.write_json(directory / f"fold_{fold}.json", saved)
                    break
            if len(folds) == len(legacy.DATASETS[dataset][0]):
                result = {
                    **legacy.summarize(folds),
                    "stop_reason": "four_folds_complete",
                    "fixed_config": str(directory / "config.json"),
                }
                legacy.write_json(directory / "result.json", result)
                study.tell(trial.number, result["source_val"])
            else:
                legacy.write_json(directory / "transition.json", {
                    "state": "superseded", "completed_folds": [fold["fold"] for fold in folds],
                    "successor_run": f"{OLD_NAME}_low_contrast",
                })
        legacy.refresh(dataset, study, root, "superseded", gpu=legacy.GPU_FOR_DATASET[dataset],
                       stop_reason="contrastive_space_updated")


if __name__ == "__main__":
    drain()
    reconcile()
    progress("ready")
