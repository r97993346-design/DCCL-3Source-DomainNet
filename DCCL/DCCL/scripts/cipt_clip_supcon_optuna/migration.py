"""Independent warm handoff from the original CLIP SupCon Optuna studies."""
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import uuid
from pathlib import Path

import optuna

import run as runner
from protocol import (
    CLIP, CODE, DATA, DATASETS, LR_POLICY, OLD_OUTPUT, OLD_RUN_NAME, OUTPUT, PAUSE, PROTOCOL_ARGS,
    SEARCH_KEYS, STEPS, budget_for, git, hparams_for, read_json, read_rows, seed_for, signature,
    space_for, summarize, validate_fold, write_json,
)

sys.path.insert(0, "/media/hooasia/data/global-optuna-registry")
from global_registry import Registry, Unavailable, configuration, digest, model_fingerprint
from registry_gate import Occupied, TrialGate

SERVER = "school-hooasia"
PEER = "autodl-39636"
REGISTRY = "git@github.com:r97993346-design/DCCL-3Source-DomainNet.git"


def historical_trials(dataset):
    path = OLD_OUTPUT / dataset / "study.sqlite3"
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=15) as db:
        db.execute("BEGIN")
        rows = db.execute(
            "SELECT t.trial_id,t.number,t.state,v.value FROM trials t "
            "LEFT JOIN trial_values v ON t.trial_id=v.trial_id AND v.objective=0 "
            "ORDER BY t.number"
        ).fetchall()
        trials = []
        for tid, number, state, value in rows:
            params = {
                key: json.loads(dist)["attributes"]["choices"][int(index)]
                for key, index, dist in db.execute(
                    "SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=?", (tid,)
                )
            }
            trials.append(dict(number=number, state=state, params=params, objective=value))
        return trials


def old_results(dataset, item):
    path = OLD_OUTPUT / dataset / "trials" / f"trial_{item['number']:03d}"
    params = item["params"]
    old = read_json(path / "result.json")
    if read_json(path / "config.json") != hparams_for(params):
        raise ValueError("historical configuration differs")
    folded = []
    for fold in range(4):
        row = read_json(path / f"fold_{fold}.json")
        raw = Path(row["log"])
        expected = path / f"env{fold}"
        attempt = raw.parents[3]
        if attempt.parent != expected or not raw.resolve().is_relative_to(path.resolve()):
            raise ValueError("historical results outside matching trial")
        check = runner.read_attempt(attempt, dataset, fold, params)
        if row != check or old["folds"][fold] != check:
            raise ValueError("historical fold markers differ from logs")
        folded.append(check)
    scored = summarize(folded)
    for metric in ("source_val", "iid", "test_in", "test_out"):
        if not math.isclose(old[metric], scored[metric], abs_tol=1e-14, rel_tol=0):
            raise ValueError("historical aggregate differs")
    if not math.isclose(item["objective"], scored["source_val"], abs_tol=1e-14, rel_tol=0):
        raise ValueError("historical objective differs from source validation")
    return folded


def copy_history(root, number, dataset, item, folds):
    dest = root / "trials" / f"trial_{number:03d}"
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "config.json", hparams_for(item["params"]))
    copies = []
    for fold in folds:
        log = Path(fold["log"])
        attempt = log.parents[3]
        out = dest / f"env{fold['fold']}" / attempt.name
        target = out / log.relative_to(attempt)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log, target)
        for name in ("learning_rates.jsonl", "console.log", "exit_code", "elapsed_seconds"):
            shutil.copyfile(attempt / name, out / name)
        copied = dict(fold, log=str(target), learning_rates=str(out / "learning_rates.jsonl"))
        write_json(dest / f"fold_{fold['fold']}.json", copied)
        copies.append(copied)
    write_json(dest / "result.json", dict(summarize(copies), stop_reason="verified_history",
        fixed_config=str(dest / "config.json"), source_search=OLD_RUN_NAME,
        source_trial=item["number"]))


def migrate(dataset, root, study, sampler):
    old = historical_trials(dataset)
    if any(item["state"] == "RUNNING" for item in old):
        raise ValueError("source study still has a RUNNING trial")
    space = space_for(dataset)
    distributions = {key: optuna.distributions.CategoricalDistribution(values) for key, values in space.items()}
    seen = {signature(t.params, space): t.number for t in study.get_trials(deepcopy=False)
            if t.state == optuna.trial.TrialState.COMPLETE and set(t.params) == set(space)}
    ids = {t.user_attrs.get("source_trial"): t.number for t in study.get_trials(deepcopy=False)
           if t.user_attrs.get("source_search") == OLD_RUN_NAME}
    imported, excluded = [], []
    for item in old:
        number = item["number"]
        if item["state"] != "COMPLETE":
            excluded.append(dict(item, reason="not_complete"))
            continue
        if dataset == "VLCS":
            excluded.append(dict(item, reason="old_initial_lr_outside_new_space", reference_only=True))
            continue
        try:
            sig = signature(item["params"], space)
            folds = old_results(dataset, item)
        except (ValueError, KeyError, OSError, TypeError) as error:
            excluded.append(dict(item, reason=str(error)))
            continue
        if number in ids:
            imported.append({"old_trial": number, "new_trial": ids[number]})
            continue
        if sig in seen:
            excluded.append(dict(item, reason="duplicate_configuration"))
            continue
        new_id = len(study.get_trials(deepcopy=False))
        copy_history(root, new_id, dataset, item, folds)
        study.add_trial(optuna.trial.create_trial(params=item["params"], distributions=distributions,
            value=item["objective"], user_attrs={"origin":"imported", "candidate_group":"verified_history",
            "source_search":OLD_RUN_NAME, "source_trial":number}))
        assert study.get_trials(deepcopy=False)[-1].number == new_id
        sampler.reserve(item["params"])
        seen[sig] = new_id
        imported.append({"old_trial": number, "new_trial": new_id})
    report = {"source_search": OLD_RUN_NAME, "dataset": dataset, "imported": imported,
              "excluded": excluded, "historical_complete": sum(t["state"]=="COMPLETE" for t in old)}
    write_json(root / "migration.json", report)
    return report


def peer_ready(registry):
    # Inventory snapshots cannot attest that admissions use the atomic registry.
    # The peer controller must publish this receipt after its guarded rollout.
    with tempfile.TemporaryDirectory() as path:
        if not registry.checkout(path):
            raise Unavailable("global registry branch missing")
        root = Path(path)
        p = root / "participants" / f"{PEER}.json"
        if not p.is_file():
            return False
        proof = read_json(p)
        if (proof.get("server_id") != PEER or proof.get("schema_version") != 1
                or proof.get("claims_before_launch") is not True
                or proof.get("inventory_reconciled") is not True
                or not proof.get("verified_claim_key")):
            return False
        claim_key = Path(proof["verified_claim_key"])
        if claim_key.is_absolute() or ".." in claim_key.parts or not (root / claim_key).is_file():
            return False
        receipt = read_json(root / claim_key)
        effective = receipt.get("effective_config") or {}
        if receipt.get("server_id") != PEER or not receipt.get("owner_token"):
            return False
        return (claim_key == registry.key(receipt["dataset"], receipt["config_hash"])
                and effective.get("dataset") == receipt["dataset"]
                and digest(effective) == receipt["config_hash"])


def sha_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else stream_sha(handle)


def stream_sha(handle):
    h = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def dataset_hash(dataset):
    root = DATA / DATASETS[dataset][2]
    allowed = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed)
    if not files:
        raise ValueError("empty training dataset")
    return digest([[p.relative_to(root).as_posix(), sha_file(p)] for p in files])


def identity(dataset):
    # Historical rows come from the immutable training implementation inherited by this profile.
    old = next(t for t in historical_trials(dataset) if t["state"] == "COMPLETE")
    proof = old_results(dataset, old)[0]
    sample = read_rows(Path(proof["log"]))[0]
    args = dict(sample["args"], batch_size=sample["hparams"]["batch_size"])
    return dict(args=args, hparams=sample["hparams"], data=dataset_hash(dataset),
                weights=sha_file(CLIP), implementation=model_fingerprint(CODE))


def config_for(dataset, params, source):
    hp = dict(source["hparams"], **hparams_for(params))
    return configuration(dataset, source["data"], source["weights"], source["implementation"],
                         source["args"], hp, [seed_for(dataset, i) for i in range(4)], LR_POLICY)


def log_registry(root, event, **fields):
    record = {"event": event, **fields}
    with (root / "registry_events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    runner.emit(event, dataset=root.name, **fields)


def tell_complete(study, number, result):
    study.tell(number, result["source_val"])


def archive_location(root, number):
    name = OUTPUT.name
    branch = "results/" + name.replace("_", "-")
    manifest = f"DCCL/DCCL/analysis_results/{name}/{root.name}/trials/trial_{number:03d}/registry_manifest.json"
    return branch, manifest


def request_sync():
    import subprocess
    subprocess.run(["systemctl", "--user", "start", "--no-block", "cipt-github-results-sync.service"],
                   check=False, timeout=15)


def guarded_trial(dataset, root, study, trial, params, gpu, env, lock_fd, gate, config):
    number = trial.number
    output = root / "trials" / f"trial_{number:03d}" / "result.json"
    with gate.eligible(trial, config) as reservation:
        def on_fold():
            markers = sorted(int(p.stem.split("_")[-1]) for p in output.parent.glob("fold_*.json"))
            while True:
                try:
                    _, current = gate.fold_finished(reservation, markers)
                    if not current or current["status"] != "RUNNING" or current["owner_token"] != reservation[1]:
                        raise ValueError("loss of global ownership before next fold")
                    request_sync()
                    return
                except Unavailable:
                    time.sleep(runner.POLL)

        log_registry(root, "registry_claim_confirmed", trial=number, config_hash=digest(config))
        result = runner.finish_trial(root, dataset, study, number, params, gpu, env, lock_fd, on_fold=on_fold)
        if "error" in result:
            gate.registry.progress(*reservation, status="FAILED", error="fold_failed")
            study.tell(number, state=optuna.trial.TrialState.FAIL)
            return
        for fold in result["folds"]:
            for row in read_rows(Path(fold["log"])):
                args = dict(row["args"], batch_size=row["hparams"]["batch_size"])
                logged = configuration(dataset, config["dataset_version"], config["weights_sha256"],
                    config["implementation_sha256"], args, row["hparams"], config["fold_seeds"], LR_POLICY)
                if logged != config:
                    raise ValueError("actual configuration differs from the globally claimed one")
        write_json(output, dict(result, config_hash=digest(config)))
        on_fold()
        branch, path = archive_location(root, number)
        while True:
            try:
                changed, record = gate.publish_complete(reservation, branch, path)
                if not record or record["status"] != "COMPLETE":
                    raise Unavailable("global completion is unconfirmed")
                tell_complete(study, number, result)
                log_registry(root, "registry_result_published", trial=number, config_hash=digest(config),
                             branch=branch, path=path)
                return
            except (Unavailable, FileNotFoundError):
                time.sleep(runner.POLL)


def reuse(root, dataset, trial, params, config, registry, record):
    fingerprint = digest(config)
    if record["config_hash"] != fingerprint or record["effective_config"] != config:
        raise ValueError("different configurations cannot be reused")
    branch, path = record["result_branch"], record["result_path"]
    manifest = registry.published(branch, path, fingerprint)
    destination = root / "trials" / f"trial_{trial.number:03d}"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "config.json", hparams_for(params))
    folds = []
    with tempfile.TemporaryDirectory() as temp:
        if not registry.checkout(temp, f"refs/heads/{branch}"):
            raise Unavailable("published branch missing")
        for fold in manifest["folds"]:
            index = fold["fold"]
            out = destination / f"env{index}" / "attempt_001"
            log = out / "raw" / dataset / "external" / "results.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            for key, target in (("results", log), ("learning_rates", out / "learning_rates.jsonl")):
                source = Path(temp) / fold[key]["path"]
                if sha_file(source) != fold[key]["sha256"]:
                    raise ValueError("published log changed since validation")
                shutil.copyfile(source, target)
            checked_rows = read_rows(log)
            for row in checked_rows:
                args, hp = row["args"], row["hparams"]
                comparable = configuration(dataset, config["dataset_version"], config["weights_sha256"],
                    config["implementation_sha256"], dict(args, batch_size=hp["batch_size"]), hp,
                    config["fold_seeds"], LR_POLICY)
                if comparable != config:
                    raise ValueError("published effective training configuration differs")
                args.update(data_dir=str(DATA), cipt_clip_path=str(CLIP))
            row = validate_fold(checked_rows, dataset, index, params)
            trace = read_rows(out / "learning_rates.jsonl")
            if len(trace) != STEPS or any(t["step"] != i or not math.isclose(
                t["lr_used"], runner.cosine_lr(i, params["initial_lr"]), rel_tol=1e-7, abs_tol=1e-16
            ) or not math.isclose(t["lr_after_scheduler"], runner.cosine_lr(i+1, params["initial_lr"]),
                                  rel_tol=1e-7, abs_tol=1e-16) for i,t in enumerate(trace)):
                raise ValueError("published scheduler trace failed validation")
            row.update(log=str(log), learning_rates=str(out/"learning_rates.jsonl"),
                       actual_prompt_length=params["prompt_length"])
            write_json(destination / f"fold_{index}.json", row)
            folds.append(row)
    result = dict(summarize(folds), config_hash=fingerprint, stop_reason="verified_global_reuse",
                  external_server=record["server_id"], result_branch=branch, result_path=path)
    if any(not math.isclose(result[k], record[k], rel_tol=0, abs_tol=1e-12)
           for k in ("source_val", "iid", "test_out")):
        raise ValueError("published objective disagrees with global registry")
    trial.set_user_attr("origin", "external")
    write_json(destination / "result.json", result)
    return result


def register_history(dataset, root, study, registry, source):
    marker = root / "registry_migration.json"
    if marker.exists():
        return
    imported = [t for t in study.get_trials(deepcopy=False) if t.user_attrs.get("origin") == "imported"]
    pending = []
    for trial in imported:
        config = config_for(dataset, trial.params, source)
        key = digest(config)
        result_path = root/"trials"/f"trial_{trial.number:03d}"/"result.json"
        result = read_json(result_path)
        if result.get("config_hash") != key:
            write_json(result_path, dict(result, config_hash=key))
        pending.append((trial.number, config, key))
    request_sync()
    recorded = []
    for number, config, key in pending:
        token = digest({"server": SERVER, "study": study.study_name, "configuration": key})
        won, remote = registry.claim(config, server_id=SERVER, study=study.study_name,
            trial_id=number, commit=git("rev-parse", "HEAD"), owner_token=token)
        if won:
            ref, path = archive_location(root, number)
            registry.progress(registry.key(dataset, key), token, completed_folds=list(range(4)))
            _, remote = registry.complete(registry.key(dataset, key), token, ref, path,
                lambda m: {k: sum(f[k] for f in m["folds"])/4 for k in ("source_val","iid","test_out")})
            if not remote or remote["status"] != "COMPLETE":
                raise Unavailable("historical completion could not be confirmed")
        recorded.append({"trial": number, "config_hash": key, "registry_state": remote["status"],
                         "registry_server": remote["server_id"]})
    write_json(marker, {"dataset": dataset, "verified_history_in_registry": recorded})


def after_boundary(dataset, root, study, sampler, gpu, gpu_uuid, lock_fd):
    while not runner.previous_study_drained(OLD_RUN_NAME, dataset):
        runner.refresh(dataset, study, root, "waiting_for_previous_trial", gpu=gpu)
        time.sleep(runner.POLL)
    report = migrate(dataset, root, study, sampler)
    registry = Registry(REGISTRY)
    gate = TrialGate(registry, SERVER, study.study_name, git("rev-parse", "HEAD"))
    source = None
    env = runner.environment_for(gpu_uuid)
    prior_budget = budget_for(OLD_OUTPUT, dataset)
    prior_excluded_budget = report["historical_complete"] if dataset == "VLCS" else 0
    while True:
        done = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if any(read_json(root/"trials"/f"trial_{t.number:03d}"/"result.json")["iid"] >= DATASETS[dataset][1]
               for t in done):
            return runner.refresh(dataset, study, root, "complete", gpu=gpu, reason="iid_target_reached")
        total = budget_for(OUTPUT, dataset) if (OUTPUT/"trial_budgets.json").exists() else prior_budget
        if len(done) + prior_excluded_budget >= total:
            return runner.refresh(dataset, study, root, "complete", gpu=gpu, reason="completed_trial_budget")
        pending = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
        if not pending and PAUSE.exists():
            return runner.refresh(dataset, study, root, "paused", gpu=gpu, reason="requested_after_current_trial")
        try:
            if not peer_ready(registry):
                raise Unavailable("rental controller has no verified atomic-claim receipt")
            # Reading the current registry is required even for a local resumed trial.
            if source is None:
                source = identity(dataset)
            register_history(dataset, root, study, registry, source)
            if pending:
                frozen, = pending
                trial = optuna.trial.Trial(study, frozen._trial_id)
                if set(trial.params) != set(space_for(dataset)):
                    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
                    continue
            else:
                if not sampler.remaining:
                    return runner.refresh(dataset, study, root, "complete", gpu=gpu, reason="search_space_exhausted")
                trial = study.ask()
                for name, choices in space_for(dataset).items():
                    trial.suggest_categorical(name, choices)
                sampler.reserve(trial.params)
                sampler.checkpoint_rng()
            params = trial.params
            config = config_for(dataset, params, source)
            try:
                guarded_trial(dataset, root, study, trial, params, gpu, env, lock_fd, gate, config)
            except Occupied as occupied:
                record = occupied.record
                if record["status"] == "COMPLETE":
                    own = root / "trials" / f"trial_{trial.number:03d}" / "result.json"
                    if record["server_id"] == SERVER and own.is_file():
                        result = read_json(own)
                        if result.get("config_hash") != digest(config) or result["source_val"] != record["source_val"]:
                            raise ValueError("locally recorded result and registry disagree")
                    else:
                        result = reuse(root, dataset, trial, params, config, registry, record)
                    tell_complete(study, trial.number, result)
                    log_registry(root, "registry_result_reused", trial=trial.number, config_hash=digest(config),
                                 server=record["server_id"])
                else:
                    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
                    log_registry(root, "registry_occupied", trial=trial.number, config_hash=digest(config),
                                 owner=record["server_id"], state=record["status"])
        except (Unavailable, FileNotFoundError) as error:
            runner.refresh(dataset, study, root, "waiting_for_registry", gpu=gpu, reason=str(error))
            time.sleep(runner.POLL)
