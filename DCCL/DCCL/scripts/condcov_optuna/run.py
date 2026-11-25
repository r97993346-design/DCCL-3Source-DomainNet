"""Run one persistent, serial CondCov study per dataset on fixed GPUs."""

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
from threading import Lock

import optuna
import yaml
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState

import history
from protocol import (
    BASE, BRANCH, CLIP, CODE, DATA, DATASETS, DIAG_STEPS, EVAL_STEPS, FIXED_ARGS, INITIAL_BUDGET,
    MAX_NEW, MODEL_COMMIT, N_JOBS, OUTPUT, SAMPLER_SEED, SPACE, command, fixed_hparams, git, hparams_for,
    read_json, read_rows, require, same_number, seed_for, signature, summarize, validate_fold,
    write_json, write_text,
)
from reports import numbers, records_for, refresh
from sampler import UniqueTPE


GPU_FOR_DATASET = {"PACS": 0, "VLCS": 1, "OfficeHome": 1}
MIN_RAM_KIB = 10 * 1024 * 1024
RESOURCE_POLL_SECONDS = 60
RAM_ADMISSION = Lock()
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
DISTRIBUTIONS = {key: CategoricalDistribution(choices) for key, choices in SPACE.items()}


class FailedFold(ValueError):
    """A training attempt exhausted its retries."""


def emit(event, **fields):
    record = {"time": datetime.now().astimezone().isoformat(), "event": event, **fields}
    print(json.dumps(record, ensure_ascii=False), flush=True)


def gpu_uuids():
    # NVIDIA's physical index is used only to obtain its stable CUDA identifier.
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True, timeout=20,
    )
    identities = {}
    for line in text.splitlines():
        index, uuid = (part.strip() for part in line.split(","))
        identities[int(index)] = uuid
    require(set(GPU_FOR_DATASET.values()) <= identities.keys(), "assigned GPUs were not found")
    return identities


def environment_for(uuid):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")


def available_ram_kib():
    with Path("/proc/meminfo").open() as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    raise ValueError("MemAvailable is missing from /proc/meminfo")


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


def preflight(identities):
    dest = OUTPUT / "preflight"
    dest.mkdir(parents=True, exist_ok=True)
    env = environment_for(identities[0])
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


def recover_trial(study):
    running = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    require(len(running) <= 1, "multiple active trials in one study")
    if running:
        trial = running[0]
        if set(trial.params) == set(SPACE):
            return trial.number, trial.params
        study.tell(trial.number, state=TrialState.FAIL)
    return None


def suggest_trial(trial, sampler):
    require(sampler.remaining, "categorical search space exhausted")
    trial.set_user_attr("origin", "new")
    params = {name: trial.suggest_categorical(name, choices) for name, choices in SPACE.items()}
    require(signature(params) in sampler.remaining, "sampler proposed an already reserved trial")
    sampler.reserve(params)
    sampler.checkpoint_rng()
    return params


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


def start_attempt(attempt, dataset, fold, trial, config, gpu, env, lock_fd):
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
    status_path = OUTPUT / dataset / "search_status.json"
    waiting = False
    with log.open("wb") as stream:
        while True:
            # Only serialize admission. Release the lock while this fold is running.
            with RAM_ADMISSION:
                available = available_ram_kib()
                if available > MIN_RAM_KIB:
                    child = subprocess.Popen(["bash", str(wrapper)], env=env, stdout=stream,
                                             stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
                    break
            if not waiting:
                state = read_json(status_path)
                write_json(status_path, dict(state, state="waiting_for_ram",
                                             mem_available_kib=available, min_ram_kib=MIN_RAM_KIB))
                emit("ram_wait", dataset=dataset, gpu=gpu, trial=trial, fold=fold,
                     mem_available_kib=available)
                waiting = True
            time.sleep(RESOURCE_POLL_SECONDS)
        state = read_json(status_path)
        write_json(status_path, dict(state, state="running", mem_available_kib=available,
                                     min_ram_kib=MIN_RAM_KIB))
        emit("fold_started", dataset=dataset, trial=trial, fold=fold, gpu=gpu, attempt=attempt.name,
             seed=seed_for(dataset, fold), mem_available_kib=available, wrapper_pid=child.pid)
        child.wait()


def run_fold(root, dataset, fold, trial, params, gpu, env, lock_fd):
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
        start_attempt(attempt, dataset, fold, trial, root / "config.json", gpu, env, lock_fd)
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
    raise FailedFold(f"fold {fold} failed twice; see {attempts[-1] / 'console.log'}")


def finish_trial(root, dataset, study, number, params, gpu, env, lock_fd):
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
            value = run_fold(trial, dataset, fold, number, params, gpu, env, lock_fd)
        except FailedFold as error:
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


def run_study(dataset, root, study, sampler, gpu, uuid, lock_fd):
    env = environment_for(uuid)

    def objective(trial):
        params = suggest_trial(trial, sampler)
        result = finish_trial(root, dataset, study, trial.number, params, gpu, env, lock_fd)
        if "error" in result:
            raise FailedFold(result["error"])
        return result["source_val"]

    while True:
        reason = stop_reason(dataset, study, root)
        if reason:
            status = refresh(dataset, study, root, "stopped", gpu=gpu, stop_reason=reason)
            emit("study_stopped", dataset=dataset, reason=reason, new=status["new_complete"])
            return
        pending = recover_trial(study)
        if pending is None:
            study.optimize(objective, n_trials=1, n_jobs=N_JOBS, catch=(FailedFold,))
        else:
            number, params = pending
            result = finish_trial(root, dataset, study, number, params, gpu, env, lock_fd)
            if "error" in result:
                study.tell(number, state=TrialState.FAIL)
            else:
                study.tell(number, result["source_val"])
        record = records_for(study, root)[-1]
        if record["state"] == "COMPLETE":
            emit("trial_completed", dataset=dataset, trial=record["trial_id"], params=record["params"],
                 source_val=record["source_val"], iid=record["iid"], test_out=record["test_out"])
        else:
            emit("trial_failed", dataset=dataset, trial=record["trial_id"], error=record.get("error"))
        refresh(dataset, study, root, "queued", gpu=gpu)


def schedule(studies, identities, lock_fd):
    with ThreadPoolExecutor(max_workers=len(studies)) as pool:
        futures = []
        for dataset, (root, study, sampler) in studies.items():
            gpu = GPU_FOR_DATASET[dataset]
            futures.append(pool.submit(run_study, dataset, root, study, sampler, gpu, identities[gpu], lock_fd))
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
            "training_seeds": {ds: [seed_for(ds, f) for f in range(4)] for ds in DATASETS},
            "scheduler": {"fixed_gpus": GPU_FOR_DATASET, "n_jobs_per_study": N_JOBS,
                          "ram_min_kib_exclusive": MIN_RAM_KIB}}
    stored = OUTPUT / "plan.json"
    if stored.exists():
        require(read_json(stored) == plan, "persisted search protocol changed")
    else:
        write_json(stored, plan)


def main():
    os.chdir(CODE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["--preflight"]:
        preflight(gpu_uuids())
        return
    require(not sys.argv[1:], "use no arguments or --preflight")
    with (BASE / ".condcov_optuna.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        create_plan()
        identities = gpu_uuids()
        scheduler = {"state": "ram_only_admission", "fixed_gpus": GPU_FOR_DATASET,
                     "min_ram_kib_exclusive": MIN_RAM_KIB, "n_jobs_per_study": N_JOBS}
        write_json(OUTPUT / "handoff.json", scheduler)
        emit("scheduler_started", **scheduler)
        for dataset, (_, target, _) in DATASETS.items():
            state = OUTPUT / dataset / "search_status.json"
            previous = read_json(state) if state.exists() else {}
            write_json(state, dict(previous, state="initializing", iid_target=target,
                                   gpu=GPU_FOR_DATASET[dataset], n_jobs=N_JOBS))
        preflight(identities)
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
        schedule(studies, identities, lock.fileno())
        final = {dataset: read_json(root / "search_status.json") for dataset, (root, _, _) in studies.items()}
        write_json(OUTPUT / "final_status.json", final)
        emit("all_studies_stopped", studies=final)


if __name__ == "__main__":
    main()
