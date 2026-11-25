"""Wait for the active grids, then run the three persistent CondCov studies."""

import csv
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import optuna
import yaml
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState

import history
from protocol import (
    BASE, BRANCH, CLIP, CODE, DATA, DATASETS, DIAG_STEPS, EVAL_STEPS, FIXED_ARGS, INITIAL_BUDGET,
    MAX_NEW, MODEL_COMMIT, OUTPUT, SAMPLER_SEED, SPACE, command, fixed_hparams, git, hparams_for,
    read_json, read_rows, require, same_number, seed_for, signature, summarize, validate_fold,
    write_json, write_text,
)
from reports import numbers, records_for, refresh
from sampler import UniqueTPE


MIN_FREE_MIB = 10 * 1024
RESOURCE_POLL_SECONDS = 60
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
DISTRIBUTIONS = {key: CategoricalDistribution(choices) for key, choices in SPACE.items()}


def emit(event, **fields):
    record = {"time": datetime.now().astimezone().isoformat(), "event": event, **fields}
    print(json.dumps(record, ensure_ascii=False), flush=True)


def nvidia_rows(options):
    text = subprocess.check_output(["nvidia-smi", *options, "--format=csv,noheader,nounits"],
                                   text=True, timeout=20)
    return [tuple(value.strip() for value in line.split(",")) for line in text.splitlines() if line.strip()]


def active_tmux():
    state = subprocess.run(["tmux", "list-panes", "-a", "-F", "#{session_name} #{pane_dead}"],
                           text=True, capture_output=True)
    return {line.split()[0] for line in state.stdout.splitlines() if line.endswith(" 0")}


def grid_status():
    sessions = active_tmux()
    gpu_args = []
    for _, pid in nvidia_rows(["--query-compute-apps=gpu_uuid,pid"]):
        try:
            args = (Path("/proc") / pid / "cmdline").read_bytes()
        except OSError:
            continue
        gpu_args.append(args)
    states = {}
    for folder, lock_name, total in history.GRID_SPECS:
        output = history.output_for_grid(folder)
        progress = read_json(output / "progress.json")
        lockfile = history.DATA_VOLUME / folder / f".{lock_name}.lock"
        try:
            with lockfile.open() as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = False
        except BlockingIOError:
            locked = True
        gpu_busy = any(folder.encode() in args for args in gpu_args)
        has_outputs = False
        if progress["completed_folds"] == total:
            with (output / "folds.csv").open(encoding="utf-8") as folds:
                fold_rows = list(csv.DictReader(folds))
            has_outputs = (len(fold_rows) == total and (output / "report.md").stat().st_size > 0
                           and (output / "summary.csv").stat().st_size > 0
                           and all((output / r["Log"]).is_file() for r in fold_rows))
        states[folder] = {
            "completed": progress["completed_folds"], "total": total,
            "failure_count": len(progress.get("failed", [])), "queue_active": lock_name in sessions,
            "lock_held": locked, "gpu_process_active": gpu_busy, "results_verified": has_outputs,
        }
    return states


def wait_for_grid_completion():
    previous = None
    while True:
        try:
            states = grid_status()
            ready = all(s["completed"] == s["total"] and s["results_verified"] and s["failure_count"] == 0
                        and not any(s[k] for k in ("queue_active", "lock_held", "gpu_process_active"))
                        for s in states.values())
            snapshot = {"state": "ready" if ready else "waiting_for_grids", "grids": states}
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            ready = False
            snapshot = {"state": "waiting_for_grid_verification", "error": str(error)}
        write_json(OUTPUT / "handoff.json", snapshot)
        if snapshot != previous:
            emit("grid_handoff", **snapshot)
            previous = snapshot
        if ready:
            return
        time.sleep(RESOURCE_POLL_SECONDS)


def environment_for(uuid):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")


def resources(index):
    index, uuid, free = nvidia_rows(["-i", str(index), "--query-gpu=index,uuid,memory.free"])[0]
    with Path("/proc/meminfo").open() as stream:
        ram = int(next(line for line in stream if line.startswith("MemAvailable:")).split()[1]) / 1024
    return int(index), uuid, int(free), ram


def wait_resources(index, has_work=lambda: True):
    announced = False
    while has_work():
        try:
            i, uuid, gpu_free, ram = resources(index)
            if i == index and gpu_free > MIN_FREE_MIB and ram > MIN_FREE_MIB:
                return uuid
            if not announced:
                emit("resource_wait", gpu=index, free_mib=gpu_free, ram_mib=ram)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            if not announced:
                emit("resource_query_error", gpu=index, error=str(error))
        announced = True
        time.sleep(RESOURCE_POLL_SECONDS)
    return None


def inspect_show(log, dataset, params):
    text = ANSI.sub("", log.read_text(encoding="utf-8"))
    try:
        args, hp = text.split("HParams:", 1)
        args = args.split("Args:", 1)[1]
    except (IndexError, ValueError) as error:
        raise ValueError(f"missing resolved configuration in {log}") from error
    expected_args = dict(FIXED_ARGS, dataset=dataset, source_only_selection=True, deterministic=True)
    for section, expected in ((args, expected_args), (hp, hparams_for(params))):
        for key, value in expected.items():
            match = re.search(rf"^\s*{re.escape(key)}:\s*([^\r\n]+)", section, flags=re.M)
            require(match is not None, f"preflight missing {key}")
            raw = match.group(1).strip()
            if isinstance(value, float):
                parsed = float(raw)
            else:
                parsed = None if raw == "None" else yaml.safe_load(raw)
            require(parsed == value, f"preflight mismatch {key}: got {parsed!r}, expected {value!r}")


def preflight():
    dest = OUTPUT / "preflight"
    dest.mkdir(parents=True, exist_ok=True)
    _, uuid, _, _ = resources(0)
    env = environment_for(uuid)
    for dataset in DATASETS:
        for candidate in range(3):
            params = {key: choices[candidate] for key, choices in SPACE.items()}
            run = dest / dataset / f"prompt_{params['prompt_length']}"
            run.mkdir(parents=True, exist_ok=True)
            proof = run / "verified.json"
            if proof.exists():
                continue
            config = run / "config.json"
            write_json(config, hparams_for(params))
            log = run / "show.log"
            with log.open("wb") as stream:
                subprocess.run(command(dataset, 0, "preflight", config, run / "raw") + ["--show"],
                               env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=120)
            inspect_show(log, dataset, params)
            write_json(proof, {"dataset": dataset, "params": params, "hparams": hparams_for(params)})
            emit("preflight_verified", dataset=dataset, params=params)


def study_for(dataset, root):
    sampler = UniqueTPE(root / "sampler_rng.json")
    storage = optuna.storages.RDBStorage(f"sqlite:///{root / 'study.sqlite3'}",
                                        engine_kwargs={"connect_args": {"timeout": 60}})
    study = optuna.create_study(study_name=f"{dataset}_condcov_joint_seed1", direction="maximize",
                               sampler=sampler, pruner=optuna.pruners.NopPruner(), storage=storage,
                               load_if_exists=True)
    return study, sampler


def import_observations(study, sampler, scanned):
    seen = {signature(trial.params) for trial in study.get_trials(deepcopy=False)
            if set(trial.params) == set(SPACE) and trial.user_attrs.get("origin") != "duplicate"}
    for item in scanned["imports"]:
        vector = signature(item["params"])
        if vector in seen:
            continue
        trial = optuna.trial.create_trial(params=item["params"], distributions=DISTRIBUTIONS,
                                          value=item["source_val"],
                                          user_attrs={"origin": "imported", "result": item})
        study.add_trial(trial)
        seen.add(vector)
    for vector in seen:
        sampler.reserve(dict(zip(SPACE, vector)))
    reference = scanned.get("reference")
    if reference and not study.get_trials(deepcopy=False) and signature(reference["params"]) not in seen:
        study.enqueue_trial(reference["params"], user_attrs={"origin": "new", "initial_reference": True},
                            skip_if_exists=True)


def next_trial(study, sampler):
    running = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    require(len(running) <= 1, "multiple active trials in one study")
    if running:
        trial = running[0]
        if set(trial.params) == set(SPACE):
            return trial.number, trial.params
        study.tell(trial.number, state=TrialState.FAIL)
    require(sampler.remaining, "categorical search space exhausted")
    trial = study.ask()
    trial.set_user_attr("origin", "new")
    params = {name: trial.suggest_categorical(name, choices) for name, choices in SPACE.items()}
    require(signature(params) in sampler.remaining, "sampler proposed an already reserved trial")
    sampler.reserve(params)
    sampler.checkpoint_rng()
    return trial.number, params


def read_attempt(attempt, dataset, fold, params):
    require((attempt / "exit_code").read_text().strip() == "0", "training process did not succeed")
    results = list((attempt / "raw").glob(f"{dataset}/*/results.jsonl"))
    require(len(results) == 1, f"expected one evaluation log, found {len(results)}")
    require(not any(attempt.rglob("*.pth")), "unexpected saved model weights")
    result = validate_fold(read_rows(results[0]), dataset, fold, params)
    telemetry = attempt / "optimizer.jsonl"
    samples = read_rows(telemetry)
    require([r["step"] for r in samples] == DIAG_STEPS
            and all(all(same_number(rate, 1e-5) for rate in r["lr"]) for r in samples),
            "missing optimizer samples or unexpected effective learning rate")
    console = (attempt / "console.log").read_text(encoding="utf-8")
    actual_lengths = re.findall(r"^PROMPT_CONTEXT_LENGTH (\d+)$", console, re.M)
    require(actual_lengths and all(int(value) == params["prompt_length"] for value in actual_lengths),
            "effective prompt length not verified")
    result.update(log=str(results[0]), telemetry=str(telemetry),
                  elapsed_seconds=float((attempt / "elapsed_seconds").read_text()),
                  actual_prompt_length=params["prompt_length"])
    return result


def start_attempt(attempt, dataset, fold, trial, config, env, lock_fd):
    attempt.mkdir()
    args = command(dataset, fold, trial, config, attempt / "raw", attempt / "optimizer.jsonl")
    output = shlex.quote(str(attempt))
    lines = ["#!/usr/bin/env bash", f"cd {shlex.quote(str(CODE))}", "start=$(date +%s)", "status=0",
             f"{shlex.join(args)} || status=$?", "finish=$(date +%s)",
             f"printf '%s\\n' \"$((finish-start))\" > {output}/elapsed_seconds.tmp",
             f"mv {output}/elapsed_seconds.tmp {output}/elapsed_seconds",
             f"printf '%s\\n' \"$status\" > {output}/exit_code.tmp",
             f"mv {output}/exit_code.tmp {output}/exit_code", 'exit "$status"']
    wrapper = attempt / "command.sh"
    write_text(wrapper, "\n".join(lines) + "\n")
    log = attempt / "console.log"
    with log.open("wb") as stream:
        subprocess.run(["bash", str(wrapper)], env=env, stdout=stream,
                       stderr=subprocess.STDOUT, pass_fds=(lock_fd,))


def run_fold(root, dataset, fold, trial, params, gpu, lock_fd):
    directory = root / f"env{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    attempts = sorted(directory.glob("attempt_*"))
    failures = 0
    for attempt in attempts:
        try:
            return read_attempt(attempt, dataset, fold, params)
        except (KeyError, ValueError, OSError):
            if (attempt / "exit_code").exists():
                failures += 1
    while failures < 2:
        attempt = directory / f"attempt_{len(attempts) + 1:03d}"
        uuid = wait_resources(gpu)
        emit("fold_started", dataset=dataset, trial=trial, fold=fold, gpu=gpu, attempt=attempt.name,
             seed=seed_for(dataset, fold))
        start_attempt(attempt, dataset, fold, trial, root / "config.json", environment_for(uuid), lock_fd)
        try:
            return read_attempt(attempt, dataset, fold, params)
        except (KeyError, ValueError, OSError) as error:
            log = attempt / "console.log"
            detail = {"error": str(error), "tail": log.read_text(encoding="utf-8", errors="replace")[-3500:],
                      "log": str(log)}
            write_json(attempt / "failure.json", detail)
            emit("fold_failed", dataset=dataset, trial=trial, fold=fold, attempt=attempt.name, error=str(error))
            failures += 1
            attempts.append(attempt)
            if failures < 2:
                time.sleep(RESOURCE_POLL_SECONDS)
    raise ValueError(f"fold {fold} failed twice; see {attempts[-1] / 'console.log'}")


def finish_trial(root, dataset, study, number, params, gpu, lock_fd):
    trial = root / "trials" / f"trial_{number:03d}"
    trial.mkdir(parents=True, exist_ok=True)
    config = trial / "config.json"
    write_json(config, hparams_for(params))
    result = trial / "result.json"
    if result.exists():
        return read_json(result)
    folds = []
    error_message = None
    for fold, domain in enumerate(DATASETS[dataset][0]):
        refresh(dataset, study, root, "running", gpu=gpu,
                active={"trial_id": number, "params": params, "target": domain, "completed_folds": len(folds)})
        try:
            value = run_fold(trial, dataset, fold, number, params, gpu, lock_fd)
        except ValueError as error:
            error_message = str(error)
            break
        folds.append(value)
        write_json(trial / f"fold_{fold}.json", value)
        emit("fold_completed", dataset=dataset, trial=number, **value)
    if error_message:
        output = {"folds": folds, "elapsed_seconds": sum(f["elapsed_seconds"] for f in folds),
                  "stop_reason": "failed_fold", "error": error_message, "fixed_config": str(config)}
    else:
        output = {**summarize(folds), "stop_reason": "four_folds_complete", "fixed_config": str(config)}
    write_json(result, output)
    return output


def stop_reason(dataset, study, root):
    records = records_for(study, root)
    _, new, _, iid_best, reached = numbers(records, dataset)
    if reached:
        return "target_iid_reached_by_import" if iid_best["origin"] == "imported" else "target_iid_reached"
    if new >= MAX_NEW:
        return "new_trial_budget_exhausted"
    tested = {signature(r["params"]) for r in records if set(r["params"]) == set(SPACE)
              and r["state"] in ("COMPLETE", "FAIL")}
    if len(tested) == 3 ** len(SPACE):
        return "no_untried_parameter_combinations"
    return None


def run_study(dataset, root, study, sampler, gpu, lock_fd):
    while True:
        reason = stop_reason(dataset, study, root)
        if reason:
            status = refresh(dataset, study, root, "stopped", stop_reason=reason)
            emit("study_stopped", dataset=dataset, reason=reason, new=status["new_complete"])
            return
        number, params = next_trial(study, sampler)
        result = finish_trial(root, dataset, study, number, params, gpu, lock_fd)
        if "error" in result:
            study.tell(number, state=TrialState.FAIL)
            emit("trial_failed", dataset=dataset, trial=number, error=result["error"])
        else:
            study.tell(number, result["source_val"])
            emit("trial_completed", dataset=dataset, trial=number, params=params,
                 source_val=result["source_val"], iid=result["iid"], test_out=result["test_out"])
        refresh(dataset, study, root, "queued", gpu=gpu)


def schedule(studies, lock_fd):
    jobs = Queue()
    for dataset, (root, study, sampler) in studies.items():
        reason = stop_reason(dataset, study, root)
        if reason:
            refresh(dataset, study, root, "stopped", stop_reason=reason)
        else:
            refresh(dataset, study, root, "queued")
            jobs.put((dataset, root, study, sampler))

    def worker(gpu):
        while wait_resources(gpu, lambda: not jobs.empty()):
            try:
                dataset, root, study, sampler = jobs.get_nowait()
            except Empty:
                return
            run_study(dataset, root, study, sampler, gpu, lock_fd)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu) for gpu in (0, 1)]
        for future in futures:
            future.result()


def create_plan():
    require(git("branch", "--show-current") == BRANCH, "wrong Git branch")
    require(not git("diff", "--name-only", MODEL_COMMIT, "HEAD", "--", *history.CODE_PATHS),
            "model or training protocol differs from the CondCov base")
    require(not git("status", "--porcelain", "--", *history.CODE_PATHS), "training files are not clean")
    require(CLIP.is_file() and all((DATA / info[2]).is_dir() for info in DATASETS.values()), "data/CLIP missing")
    plan = {"code_commit": git("rev-parse", "HEAD"), "model_commit": MODEL_COMMIT, "data": str(DATA.resolve()),
            "clip": str(CLIP.resolve()), "clip_size": CLIP.stat().st_size, "clip_mtime": CLIP.stat().st_mtime_ns,
            "fixed_args": FIXED_ARGS, "fixed_hparams": fixed_hparams(),
            "space": {k: list(v) for k, v in SPACE.items()}, "evaluation_steps": EVAL_STEPS,
            "sampler": {"name": "TPESampler", "multivariate": False, "n_startup_trials": 5,
                        "n_ei_candidates": 24, "seed": SAMPLER_SEED, "exclude_seen_combinations": True},
            "pruner": "NopPruner", "objective": "mean source validation", "direction": "maximize",
            "initial_new_budget": INITIAL_BUDGET, "max_new_budget": MAX_NEW,
            "iid_targets": {ds: info[1] for ds, info in DATASETS.items()},
            "training_seeds": {ds: [seed_for(ds, f) for f in range(4)] for ds in DATASETS}}
    stored = OUTPUT / "plan.json"
    if stored.exists():
        require(read_json(stored) == plan, "persisted search protocol changed")
    else:
        write_json(stored, plan)


def main():
    os.chdir(CODE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["--preflight"]:
        preflight()
        return
    require(not sys.argv[1:], "use no arguments or --preflight")
    with (BASE / ".condcov_optuna.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        create_plan()
        for dataset, (_, target, _) in DATASETS.items():
            state = OUTPUT / dataset / "search_status.json"
            if not state.exists():
                write_json(state, {"state": "waiting_for_grids", "iid_target": target,
                                   "new_complete": 0, "imported_complete": 0})
        wait_for_grid_completion()
        preflight()
        scan = OUTPUT / "history_scan.json"
        if scan.exists():
            scanned = read_json(scan)
        else:
            scanned = history.collect()
            write_json(scan, scanned)
        studies = {}
        for dataset in DATASETS:
            root = OUTPUT / dataset
            root.mkdir(exist_ok=True)
            study, sampler = study_for(dataset, root)
            import_observations(study, sampler, scanned[dataset])
            studies[dataset] = root, study, sampler
        schedule(studies, lock.fileno())
        final = {dataset: read_json(root / "search_status.json") for dataset, (root, _, _) in studies.items()}
        write_json(OUTPUT / "final_status.json", final)
        emit("all_studies_stopped", studies=final)


if __name__ == "__main__":
    main()
