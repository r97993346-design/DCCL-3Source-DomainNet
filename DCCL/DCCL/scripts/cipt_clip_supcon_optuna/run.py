"""Run one persistent CondCov + RD + NR study per dataset."""

import fcntl
import json
import math
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
import history_seed
import yaml
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState

from protocol import (
    BASE, BASE_COMMIT, BRANCH, BUDGET, CHECK, CLIP, CODE, DATA, DATASETS, EVAL_STEPS, FIXED_ARGS,
    LR_POLICY, MODEL_COMMIT, N_JOBS, OUTPUT, PAUSE, SAMPLER_SEED, SPACE, STEPS, command, cosine_lr,
    fixed_hparams, git, hparams_for, read_json, read_rows, require, seed_for, signature,
    study_name, summarize, validate_fold, write_json, write_text,
)
from reports import refresh
from sampler import UniqueTPE

GPU_FOR_DATASET = {"PACS": 0, "VLCS": 1, "OfficeHome": 1}
MIN_RAM_KIB = 10 * 1024 * 1024
POLL = 60
SERVICE = "cipt-clip-supcon-cosine-optuna-20261126.service"
SMOKE_LOCK = Lock()
ADMISSION = Lock()
TRIAL_ADMISSION = Lock()
EVENT_LOG = Lock()
ACTIVE_GPU = {0: 0, 1: 0}
ACTIVE_DATASETS = set()
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
OOM = re.compile(r"out of memory|OutOfMemoryError|cannot allocate memory|DefaultCPUAllocator", re.I)
DISTRIBUTIONS = {key: CategoricalDistribution(choices) for key, choices in SPACE.items()}


def emit(event, **fields):
    line = json.dumps({"time": datetime.now().astimezone().isoformat(), "event": event, **fields}, ensure_ascii=False)
    with EVENT_LOG:
        print(line, flush=True)


def gpu_uuids():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                  text=True, timeout=20)
    observed = dict((int(index.strip()), uuid.strip()) for index, uuid in (l.split(",") for l in out.splitlines()))
    require(set(GPU_FOR_DATASET.values()) <= observed.keys(), "required physical GPU is unavailable")
    return observed


def environment_for(uuid):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")


def available_ram_kib():
    with Path("/proc/meminfo").open() as stream:
        return int(next(line for line in stream if line.startswith("MemAvailable:")).split()[1])


def resume_commands():
    prefix = f"cd {shlex.quote(str(CODE))} && {shlex.quote(sys.executable)} -B scripts/cipt_clip_supcon_optuna/run.py"
    return {"pause": f"{prefix} --pause", "resume": f"{prefix} --resume"}


def create_plan():
    require(git("branch", "--show-current") == BRANCH, "wrong experiment branch")
    training_paths = ("domainbed", "train_all.py", "config_cipt_pacs_ind_ablation.yaml")
    expected = {
        "DCCL/DCCL/domainbed/trainer.py", "DCCL/DCCL/domainbed/algorithms/cipt_dccl_ablation.py",
        "DCCL/DCCL/domainbed/algorithms/cipt_clip_guided_contrastive.py",
        "DCCL/DCCL/domainbed/algorithms/cipt_env_kl.py",
    }
    require(set(git("diff", "--name-only", BASE_COMMIT, "HEAD", "--", *training_paths).splitlines()) == expected
            and not git("status", "--porcelain", "--", *training_paths), "unexpected training-code changes")
    require(CLIP.is_file() and all((DATA / info[2]).is_dir() for info in DATASETS.values()), "data or CLIP missing")
    plan = {
        "code_commit": git("rev-parse", "HEAD"), "base_commit": BASE_COMMIT, "condcov_commit": MODEL_COMMIT,
        "data": str(DATA.resolve()), "clip": str(CLIP.resolve()),
        "clip_size": CLIP.stat().st_size, "clip_mtime": CLIP.stat().st_mtime_ns,
        "fixed_args": FIXED_ARGS, "fixed_hparams": fixed_hparams(), "steps": STEPS, "checkpoint_freq": CHECK,
        "replacement": True, "class_balanced": False, "lr_policy": LR_POLICY, "evaluation_steps": EVAL_STEPS,
        "gradient_diagnostics": False, "space": {k: list(v) for k, v in SPACE.items()},
        "warm_start": "three unique historical parameter sets ranked only on source validation, both LRs per set",
        "sampler": {"name": "TPESampler", "seed": SAMPLER_SEED,
        "multivariate": False, "n_startup_trials": 5, "n_ei_candidates": 24}, "pruner": "NopPruner",
        "objective": "four source-validation accuracies at source-selected checkpoints, averaged",
        "direction": "maximize", "trial_budget": BUDGET, "target_iid_threshold_stops_study": True,
        "target_iid_reference": {d: v[1] for d, v in DATASETS.items()}, "history_import": False,
        "training_seeds": {d: [seed_for(d, i) for i in range(4)] for d in DATASETS},
        "gpu_for_dataset": GPU_FOR_DATASET, "n_jobs_per_study": N_JOBS,
        "min_ram_kib_exclusive": MIN_RAM_KIB, "oom_fallback": "serialize on the affected GPU; retry the fold",
        "study_names": {d: study_name(d) for d in DATASETS}, "commands": resume_commands(),
    }
    path = OUTPUT / "protocol.json"
    if path.exists():
        require(read_json(path) == json.loads(json.dumps(plan)), "persisted CLIP SupCon protocol changed")
    else:
        write_json(path, plan)


def inspect_show(log, dataset, params):
    text = ANSI.sub("", log.read_text(encoding="utf-8"))
    args, hparams = text.split("Args:", 1)[1].split("HParams:", 1)
    for section, expected in ((args, dict(FIXED_ARGS, dataset=dataset, source_only_selection=True,
                                          deterministic=True)), (hparams, hparams_for(params))):
        for key, value in expected.items():
            match = re.search(rf"^[ \t]*{re.escape(key)}:[ \t]*([^\r\n]+)", section, flags=re.M)
            require(match, f"preflight missing {key}")
            raw = match.group(1).strip()
            actual = float(raw) if isinstance(value, float) else None if raw == "None" else yaml.safe_load(raw)
            require(actual == value, f"preflight {key}={actual!r}, expected {value!r}")


def preflight(identities):
    cases = []
    settings = []
    for dataset in DATASETS:
        for index in range(4):
            params = {name: choices[index % len(choices)] for name, choices in SPACE.items()}
            case = OUTPUT / "audit" / "preflight" / dataset / f"candidate_{index}"
            case.mkdir(parents=True, exist_ok=True)
            proof = case / "verified.json"
            if proof.exists():
                continue
            config, log = case / "config.json", case / "show.log"
            write_json(config, hparams_for(params))
            argv = command(dataset, 0, "preflight", config, case / "raw")[3:] + ["--show"]
            cases.append({"argv": argv, "log": str(log)})
            settings.append((dataset, params, log, proof))
    if cases:
        spec = OUTPUT / "audit" / "preflight" / "entry_cases.json"
        write_json(spec, cases)
        batch_log = spec.parent / "entry_resolution.log"
        with batch_log.open("wb") as stream:
            subprocess.run([sys.executable, "-B", "-u", "scripts/cipt_clip_supcon_optuna/entry_preflight.py", str(spec)],
                           env=environment_for(identities[0]), stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=900)
    for dataset, params, log, proof in settings:
        inspect_show(log, dataset, params)
        write_json(proof, {"dataset": dataset, "params": params, "hparams": hparams_for(params)})
        emit("preflight_verified", dataset=dataset, params=params)


def open_studies():
    studies = {}
    for dataset in DATASETS:
        root = OUTPUT / dataset
        root.mkdir(exist_ok=True)
        sampler = UniqueTPE(root / "sampler_rng.json")
        storage = optuna.storages.RDBStorage(f"sqlite:///{root / 'study.sqlite3'}",
                                            engine_kwargs={"connect_args": {"timeout": 60}})
        study = optuna.create_study(study_name=study_name(dataset), direction="maximize", sampler=sampler,
                                    pruner=optuna.pruners.NopPruner(), storage=storage, load_if_exists=True)
        for trial in study.get_trials(deepcopy=False):
            if set(trial.params) == set(SPACE):
                sampler.reserve(trial.params)
        studies[dataset] = root, study, sampler
    return studies


def read_attempt(attempt, dataset, fold, params, *, updates=STEPS):
    require((attempt / "exit_code").read_text().strip() == "0", "training process did not succeed")
    logs = list((attempt / "raw").glob(f"{dataset}/*/results.jsonl"))
    require(len(logs) == 1 and not any(attempt.rglob("*.pth")), "expected one result file and no model weights")
    result = validate_fold(read_rows(logs[0]), dataset, fold, params, updates=updates)
    trace_path = attempt / "learning_rates.jsonl"
    entries = read_rows(trace_path)
    require(len(entries) == updates, "missing or extra scheduler calls")
    for i, row in enumerate(entries):
        require(row["step"] == i and row["lr_used"] > 0 and
                math.isclose(row["lr_used"], cosine_lr(i, params["initial_lr"], updates),
                             rel_tol=1e-7, abs_tol=1e-16) and
                math.isclose(row["lr_after_scheduler"], cosine_lr(i+1, params["initial_lr"], updates),
                             rel_tol=1e-7, abs_tol=1e-16), f"invalid LR trace at step {i}")
    require(entries[-1]["lr_after_scheduler"] == 0.0, "scheduler did not reach zero")
    console = (attempt / "console.log").read_text(encoding="utf-8")
    prompts = re.findall(r"^PROMPT_CONTEXT_LENGTH (\\d+)$", console, re.M)
    require(prompts and all(int(n) == params["prompt_length"] for n in prompts), "wrong prompt length")
    require(f"ADAM_UPDATES {updates} COSINE_UPDATES {updates}" in console, "missing optimizer count proof")
    return dict(result, log=str(logs[0]), learning_rates=str(trace_path), actual_prompt_length=params["prompt_length"],
                elapsed_seconds=float((attempt / "elapsed_seconds").read_text()))


def start_attempt(attempt, dataset, fold, number, config, gpu, env, lock_fd, *, updates=STEPS):
    attempt.mkdir()
    args = command(dataset, fold, number, config, attempt / "raw", attempt / "learning_rates.jsonl", updates=updates)
    output = shlex.quote(str(attempt))
    wrapper = attempt / "command.sh"
    write_text(wrapper, "\n".join([
        "#!/usr/bin/env bash", f"cd {shlex.quote(str(CODE))}", "start=$(date +%s)", "status=0",
        f"{shlex.join(args)} || status=$?", "end=$(date +%s)",
        f"printf '%s\\n' \"$((end-start))\" > {output}/elapsed_seconds.tmp",
        f"mv {output}/elapsed_seconds.tmp {output}/elapsed_seconds",
        f"printf '%s\\n' \"$status\" > {output}/exit_code.tmp",
        f"mv {output}/exit_code.tmp {output}/exit_code", 'exit "$status"',
    ]) + "\n")
    resources = OUTPUT / "resource_limits.json"
    waiting = False
    with (attempt / "console.log").open("wb") as stream:
        while True:
            with ADMISSION:
                ram = available_ram_kib()
                limits = read_json(resources) if resources.is_file() else {}
                gpu_limit = limits.get("gpu", {}).get(str(gpu), len(GPU_FOR_DATASET))
                global_limit = limits.get("global", len(GPU_FOR_DATASET))
                if ram > MIN_RAM_KIB and ACTIVE_GPU[gpu] < gpu_limit and len(ACTIVE_DATASETS) < global_limit:
                    require(dataset not in ACTIVE_DATASETS, f"duplicate active dataset {dataset}")
                    child = subprocess.Popen(["bash", str(wrapper)], env=env, stdout=stream,
                                             stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
                    ACTIVE_GPU[gpu] += 1
                    ACTIVE_DATASETS.add(dataset)
                    break
            if not waiting:
                emit("resource_wait", dataset=dataset, trial=number, fold=fold, mem_available_kib=ram, gpu=gpu)
                waiting = True
            time.sleep(POLL)
        emit("fold_started", dataset=dataset, trial=number, fold=fold, updates=updates,
             gpu=gpu, mem_available_kib=ram, pid=child.pid)
        try:
            child.wait()
        finally:
            with ADMISSION:
                ACTIVE_GPU[gpu] -= 1
                ACTIVE_DATASETS.remove(dataset)


def adjust_resources(attempt, gpu):
    tail = (attempt / "console.log").read_text(encoding="utf-8", errors="replace")[-16000:]
    exit_code = (attempt / "exit_code").read_text().strip()
    if not OOM.search(tail) and exit_code != "137":
        return tail
    with ADMISSION:
        path = OUTPUT / "resource_limits.json"
        limits = read_json(path) if path.is_file() else {}
        if "CUDA" in tail.upper():
            limits.setdefault("gpu", {})[str(gpu)] = 1
        else:
            limits["global"] = 1
        write_json(path, limits)
    emit("reduced_concurrency", gpu=gpu, limits=limits)
    return tail


def run_fold(root, dataset, fold, number, params, gpu, env, lock_fd, *, updates=STEPS):
    out = root / f"env{fold}"
    out.mkdir(exist_ok=True)
    attempts = sorted(out.glob("attempt_*"))
    failures = 0
    for attempt in attempts:
        if (attempt / "exit_code").is_file():
            try:
                return read_attempt(attempt, dataset, fold, params, updates=updates)
            except (ValueError, KeyError, OSError):
                failures += 1
    while failures < 2:
        attempt = out / f"attempt_{len(attempts) + 1:03d}"
        start_attempt(attempt, dataset, fold, number, root / "config.json", gpu, env, lock_fd, updates=updates)
        try:
            return read_attempt(attempt, dataset, fold, params, updates=updates)
        except (ValueError, KeyError, OSError) as error:
            tail = adjust_resources(attempt, gpu)
            write_json(attempt / "failure.json", {"error": str(error), "tail": tail[-4000:]})
            emit("fold_failed", dataset=dataset, trial=number, fold=fold, error=str(error), log=str(attempt))
            attempts.append(attempt)
            failures += 1
            if failures < 2:
                time.sleep(POLL)
    raise FoldFailed(f"{dataset} fold {fold} failed twice; see {attempts[-1] / 'console.log'}")


class FoldFailed(RuntimeError):
    pass


def smoke(dataset, gpu, uuid, lock_fd):
    root = OUTPUT / "audit" / "cuda_smoke"
    root.mkdir(parents=True, exist_ok=True)
    proof = root / "verified.json"
    if proof.is_file():
        return
    params = {k: v[0] for k, v in SPACE.items()}
    params["contrastive_mix"] = .25
    params["prompt_length"] = 16
    case = root/dataset
    case.mkdir(exist_ok=True)
    write_json(case / "config.json", hparams_for(params))
    result = run_fold(case, dataset, 0, "smoke", params, gpu, environment_for(uuid), lock_fd, updates=1)
    write_json(proof, {"dataset": dataset, "params": params, "result": result, "excluded_from_studies": True})
    emit("cuda_smoke_passed", result=result)


def admit_trial(study, sampler):
    with TRIAL_ADMISSION, (BASE / ".control.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if PAUSE.exists():
            return None
        require(sampler.remaining, "categorical search space exhausted")
        trial = study.ask()
        trial.set_user_attr("origin", "new")
        params = {name: trial.suggest_categorical(name, choices) for name, choices in SPACE.items()}
        require(signature(params) in sampler.remaining, "duplicate trial parameters")
        sampler.reserve(params)
        sampler.checkpoint_rng()
        return trial.number, params


def finish_trial(root, dataset, study, number, params, gpu, env, lock_fd):
    trial = root / "trials" / f"trial_{number:03d}"
    trial.mkdir(parents=True, exist_ok=True)
    output = trial / "result.json"
    if output.exists():
        return read_json(output)
    config = trial / "config.json"
    if config.exists():
        require(read_json(config) == hparams_for(params), "persisted trial configuration changed")
    else:
        write_json(config, hparams_for(params))
    folds = []
    for fold, domain in enumerate(DATASETS[dataset][0]):
        active = {"trial_id": number, "params": params, "target": domain, "completed_folds": len(folds)}
        refresh(dataset, study, root, "draining" if PAUSE.exists() else "running", gpu=gpu, active=active)
        try:
            result = run_fold(trial, dataset, fold, number, params, gpu, env, lock_fd)
        except FoldFailed as error:
            result = {"error": str(error), "folds": folds, "stop_reason": "failed_fold"}
            write_json(output, result)
            return result
        marker = trial / f"fold_{fold}.json"
        if marker.exists():
            require(read_json(marker) == result, "persisted fold summary changed")
        else:
            write_json(marker, result)
        folds.append(result)
        emit("fold_completed", dataset=dataset, trial=number, **result)
    result = dict(summarize(folds), stop_reason="four_folds_complete", fixed_config=str(config))
    write_json(output, result)
    return result


def run_study(dataset, root, study, sampler, gpu, uuid, lock_fd):
    env = environment_for(uuid)
    while True:
        pending = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
        require(len(pending) <= 1, "more than one active Trial in a study")
        if pending:
            trial = pending[0]
            if set(trial.params) != set(SPACE):
                study.tell(trial.number, state=TrialState.FAIL)
                emit("abandoned_partial_suggestion", dataset=dataset, trial=trial.number)
                continue
            next_trial = trial.number, trial.params
        else:
            complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            if any(read_json(root/"trials"/f"trial_{t.number:03d}"/"result.json")["iid"] >= DATASETS[dataset][1]
                   for t in complete):
                return refresh(dataset, study, root, "complete", gpu=gpu, reason="iid_target_reached")
            if len(complete) >= BUDGET:
                return refresh(dataset, study, root, "complete", gpu=gpu, reason="completed_trial_budget")
            next_trial = admit_trial(study, sampler)
            if next_trial is None:
                return refresh(dataset, study, root, "paused", gpu=gpu, reason="requested_after_current_trial")
        number, params = next_trial
        result = finish_trial(root, dataset, study, number, params, gpu, env, lock_fd)
        if "error" in result:
            study.tell(number, state=TrialState.FAIL)
            emit("trial_failed", dataset=dataset, trial=number, error=result["error"])
        else:
            study.tell(number, result["source_val"])
            emit("trial_completed", dataset=dataset, trial=number, source_val=result["source_val"], iid=result["iid"])
        refresh(dataset, study, root, "queued", gpu=gpu)


def after_legacy(dataset, root, study, sampler, gpu, uuid, lock_fd):
    while not history_seed.legacy_finished(dataset):
        if PAUSE.exists():
            return refresh(dataset, study, root, "paused", gpu=gpu, reason="requested_before_handoff")
        refresh(dataset, study, root, "waiting_for_legacy_dataset", gpu=gpu)
        time.sleep(30)
    history_seed.enqueue(dataset, root, study)
    if not PAUSE.exists():
        refresh(dataset, study, root, "validating_cuda_smoke", gpu=gpu)
        with SMOKE_LOCK:
            if not PAUSE.exists():
                smoke(dataset, gpu, uuid, lock_fd)
    return run_study(dataset, root, study, sampler, gpu, uuid, lock_fd)


def control(flag):
    with (BASE / ".control.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if flag == "--pause":
            write_text(PAUSE, datetime.now().astimezone().isoformat() + "\n")
            print("Warm pause requested after the current Trials.")
        else:
            active = subprocess.check_output(["systemctl", "--user", "show", SERVICE, "-p", "ActiveState", "--value"],
                                             text=True).strip()
            require(active in ("inactive", "failed"), "wait for the current Trials to finish before resuming")
            PAUSE.unlink(missing_ok=True)
            subprocess.run(["systemctl", "--user", "start", "--no-block", SERVICE], check=True)
            print("Existing CLIP SupCon studies scheduled to resume.")


def main():
    os.chdir(CODE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] in (["--pause"], ["--resume"]):
        control(sys.argv[1])
        return
    require(sys.argv[1:] in ([], ["--prepare"]), "use no arguments, --prepare, --pause or --resume")
    with (BASE / ".clip_supcon_optuna.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        create_plan()
        identities = gpu_uuids()
        studies = open_studies()
        preflight(identities)
        if sys.argv[1:] == ["--prepare"]:
            for dataset, (root, study, _) in studies.items():
                write_json(root/"history_seed_preview.json", history_seed.seed_plan(dataset))
                if history_seed.legacy_finished(dataset):
                    history_seed.enqueue(dataset, root, study)
                    state = "ready_to_launch"
                else:
                    state = "waiting_for_legacy_dataset"
                refresh(dataset, study, root, state, gpu=GPU_FOR_DATASET[dataset])
            return
        write_json(OUTPUT / "run_status.json", {"state": "running", "gpu": GPU_FOR_DATASET,
                                                 "handoff": "independent_by_dataset"})
        with ThreadPoolExecutor(max_workers=len(studies)) as pool:
            jobs = {
                dataset: pool.submit(after_legacy, dataset, root, study, sampler, GPU_FOR_DATASET[dataset],
                                     identities[GPU_FOR_DATASET[dataset]], lock.fileno())
                for dataset, (root, study, sampler) in studies.items()
            }
            final = {dataset: job.result() for dataset, job in jobs.items()}
        write_json(OUTPUT / "run_status.json", {"state": "paused" if PAUSE.exists() else "complete", "studies": final})


if __name__ == "__main__":
    main()
