"""Report complete four-fold trials and verified individual folds."""

from optuna.trial import TrialState

from protocol import DATASETS, SPACE, budget_for, read_json, write_csv, write_json


def refresh(dataset, study, root, state, *, gpu, active=None, reason=None):
    observation = []
    trials = []
    folds = []
    for trial in study.get_trials(deepcopy=False):
        directory = root / "trials" / f"trial_{trial.number:03d}"
        result_path = directory / "result.json"
        result = read_json(result_path) if result_path.is_file() else {}
        scores = result if trial.state == TrialState.COMPLETE else {}
        params = trial.params or trial.system_attrs.get("fixed_params", {})
        row = {
            "trial_id": trial.number, "state": trial.state.name, **params,
            "started_at": trial.datetime_start.isoformat() if trial.datetime_start else None,
            "ended_at": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            "candidate_group": trial.user_attrs.get("candidate_group", "tpe" if trial.params else None),
            **{k: scores.get(k) for k in ("source_val", "iid", "test_in", "test_out", "elapsed_seconds")},
            "error": result.get("error"),
        }
        trials.append(row)
        if trial.state == TrialState.COMPLETE:
            observation.append(row)
        for path in sorted(directory.glob("fold_*.json")):
            folds.append({"trial_id": trial.number, "trial_state": trial.state.name, **params, **read_json(path)})

    best = max(observation, key=lambda r: r["source_val"], default=None)
    iid_best = max(observation, key=lambda r: r["iid"], default=None)
    goal = DATASETS[dataset][1]
    status = {"dataset": dataset, "study_name": study.study_name, "state": state, "gpu": gpu, "n_jobs": 1,
              "complete_trials": len(observation), "budget": budget_for(root.parent, dataset), "active": active, "stop_reason": reason,
              "iid_reference_target": goal, "iid_reference_reached": bool(iid_best and iid_best["iid"] >= goal),
              "best_source_val": best, "best_iid": iid_best}
    trial_fields = ["trial_id", "state", *SPACE, "source_val", "iid", "test_in", "test_out", "elapsed_seconds", "error",
                    "candidate_group", "started_at", "ended_at"]
    fold_fields = ["trial_id", "trial_state", *SPACE, "fold", "domain", "seed", "selected_step", "iid", "source_val",
                   "test_in", "test_out", "final_step", "updates", "final_iid", "final_test_out", "final_source_val",
                   "elapsed_seconds", "actual_prompt_length", "log", "learning_rates", "scheduler_updates", "evaluations"]
    write_csv(root / "trials_summary.csv", trials, trial_fields)
    write_csv(root / "folds_summary.csv", folds, fold_fields)
    write_json(root / "best_source_val.json", best)
    write_json(root / "best_iid.json", iid_best)
    write_json(root / "search_status.json", status)
    return status
