"""Select historical parameter suggestions by source validation only."""

import json
import math
import sqlite3
import time

from protocol import BASE, DATASETS, SPACE, read_json, require, signature, write_json

NAMES = ("condcov_optuna_joint_20261125", "condcov_optuna_joint_20261125_low_contrast")
ROOTS = {n: BASE.parent/n/"code/DCCL/DCCL/train_output"/n for n in NAMES}
PRIOR = ROOTS[NAMES[-1]]
HISTORY_KEYS = tuple(k for k in SPACE if k != "initial_lr")


def connection(root):
    return sqlite3.connect((root/"study.sqlite3").as_uri()+"?mode=ro", uri=True, timeout=20)


def legacy_finished(dataset):
    root = PRIOR/dataset
    state = read_json(root/"search_status.json")
    if state["state"] not in ("paused", "stopped", "complete") or state["active"] is not None:
        return False
    with connection(root) as conn:
        return not conn.execute("SELECT trial_id FROM trials WHERE state='RUNNING'").fetchone()


def rank(dataset):
    candidates = []
    for name, parent in ROOTS.items():
        root = parent/dataset
        with connection(root) as db:
            rows = db.execute("""
                SELECT t.trial_id,t.number,v.value FROM trials t JOIN trial_values v USING(trial_id)
                WHERE t.state='COMPLETE' ORDER BY t.number
            """).fetchall()
            for tid, number, source_val in rows:
                params = {key: json.loads(dist)["attributes"]["choices"][int(value)]
                          for key, value, dist in db.execute(
                              "SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=?", (tid,)
                          )}
                if set(params) != set(HISTORY_KEYS) or any(params[k] not in SPACE[k] for k in HISTORY_KEYS):
                    continue
                attr = db.execute("SELECT value_json FROM trial_user_attributes WHERE trial_id=? AND key='origin'", (tid,)).fetchone()
                if attr and json.loads(attr[0]) == "imported":
                    row = db.execute("SELECT value_json FROM trial_user_attributes WHERE trial_id=? AND key='result'", (tid,)).fetchone()
                    require(row, "historical imported result is missing")
                    result = json.loads(row[0])
                else:
                    result = read_json(root/"trials"/f"trial_{number:03d}"/"result.json")
                require(math.isfinite(source_val) and math.isclose(source_val, result["source_val"], abs_tol=1e-12)
                        and [f["fold"] for f in result["folds"]] == list(range(len(DATASETS[dataset][0])))
                        and math.isclose(source_val, sum(f["source_val"] for f in result["folds"])/4, abs_tol=1e-12),
                        "incomplete or inconsistent historical source result")
                candidates.append({"source_study_run": name, "source_trial": number,
                                   "historical_source_val": source_val, "params": params})
    candidates.sort(key=lambda r: (-r["historical_source_val"], r["source_study_run"], r["source_trial"]))
    selected, seen = [], set()
    for row in candidates:
        vector = tuple(row["params"][k] for k in HISTORY_KEYS)
        if vector not in seen:
            seen.add(vector)
            selected.append(row)
            if len(selected) == 3:
                break
    return selected


def seed_plan(dataset):
    historical = rank(dataset)
    queued = []
    for rank_index, row in enumerate(historical, 1):
        for lr in SPACE["initial_lr"]:
            params = dict(initial_lr=lr, **row["params"])
            signature(params)
            queued.append({"params": params, "historical_rank": rank_index})
    return {"dataset": dataset, "top_3_by_source_val": historical, "queue": queued,
            "historical_objectives_imported": 0, "captured_unix_seconds": int(time.time())}


def enqueue(dataset, root, study):
    path = root/"history_seed_plan.json"
    if path.exists():
        plan = read_json(path)
    else:
        plan = seed_plan(dataset)
        write_json(path, plan)
    existing = {t.number: t for t in study.get_trials(deepcopy=False)}
    for index, row in enumerate(plan["queue"]):
        if index in existing:
            require(existing[index].system_attrs.get("fixed_params") == row["params"],
                    f"{dataset} warm start queue changed")
        else:
            study.enqueue_trial(row["params"], user_attrs={
                "candidate_group": "historical_warm_start", "historical_rank": row["historical_rank"],
            }, skip_if_exists=True)
    return plan
