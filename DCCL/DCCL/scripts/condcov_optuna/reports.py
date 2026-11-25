"""Persistent experiment summaries and final source-validation analysis."""

import optuna
from optuna.trial import TrialState

from diagnostics import learning_rate_report
from protocol import DATASETS, INITIAL_BUDGET, MAX_NEW, SPACE, read_json, write_csv, write_json, write_text


def records_for(study, root):
    rows = []
    for trial in study.get_trials(deepcopy=False):
        origin = trial.user_attrs.get("origin")
        if origin == "imported":
            result = trial.user_attrs["result"]
        else:
            path = root / "trials" / f"trial_{trial.number:03d}" / "result.json"
            result = read_json(path) if path.is_file() else {}
        if origin == "duplicate":
            continue
        rows.append({"trial_id": trial.number, "origin": origin or "new", "state": trial.state.name,
                     "params": trial.params, **result})
    return rows


def numbers(records, dataset):
    complete = [r for r in records if r["state"] == "COMPLETE"]
    new = sum(r["origin"] == "new" for r in complete)
    iid_best = max(complete, key=lambda r: r["iid"], default=None)
    source_best = max(complete, key=lambda r: r["source_val"], default=None)
    reached = iid_best is not None and iid_best["iid"] >= DATASETS[dataset][1]
    return complete, new, source_best, iid_best, reached


def refresh(dataset, study, root, state, *, gpu=None, active=None, stop_reason=None):
    records = records_for(study, root)
    complete, new, source_best, iid_best, reached = numbers(records, dataset)
    status = {
        "state": state, "gpu": gpu, "active": active, "stop_reason": stop_reason,
        "initial_new_budget": INITIAL_BUDGET, "max_new_budget": MAX_NEW,
        "current_new_budget": INITIAL_BUDGET if new < INITIAL_BUDGET or reached else MAX_NEW,
        "new_complete": new, "imported_complete": len(complete) - new,
        "iid_target": DATASETS[dataset][1], "objective": "mean source validation accuracy", "direction": "maximize",
        "target_reached": reached,
    }
    write_json(root / "search_status.json", status)
    write_json(root / "best_source_val.json", source_best)
    write_json(root / "best_iid.json", iid_best)
    fields = ["trial_id", "origin", "state", *SPACE, "source_val", "iid", "test_in", "test_out",
              "elapsed_seconds", "stop_reason", "error"]
    fold_fields = ("seed", "selected_step", "iid", "source_val", "test_in", "test_out",
                   "final_step", "final_iid", "final_source_val", "elapsed_seconds", "log")
    csv_rows = []
    for rec in records:
        row = {k: rec.get(k) for k in fields if k not in SPACE}
        row.update(rec["params"])
        for fold in rec.get("folds", []):
            domain = DATASETS[dataset][0][fold["fold"]]
            row.update({f"{domain}_{key}": fold.get(key) for key in fold_fields})
        csv_rows.append(row)
    fields.extend(f"{domain}_{key}" for domain in DATASETS[dataset][0] for key in fold_fields)
    write_csv(root / "trials_summary.csv", csv_rows, fields)
    by_source = sorted(complete, key=lambda r: (-r["source_val"], r["trial_id"]))
    write_csv(root / "ranking.csv", [{k: r.get(k) for k in fields} for r in sorted(
        (row for row in csv_rows if row["state"] == "COMPLETE"),
        key=lambda r: (-r["source_val"], r["trial_id"]),
    )], fields)
    learning_rate_report(dataset, root / "learning_rate_diagnosis.md", complete)
    if stop_reason:
        importance = {}
        varied = all(len({r["params"][k] for r in complete}) > 1 for k in SPACE)
        if len(complete) >= 10 and varied:
            try:
                evaluator = optuna.importance.FanovaImportanceEvaluator(seed=42)
                importance = optuna.importance.get_param_importances(study, evaluator=evaluator)
                note = "Exploratory fANOVA estimates for the source-validation objective."
            except (ValueError, RuntimeError, ImportError) as error:
                note = f"fANOVA unavailable ({error})."
        else:
            note = "Importance is descriptive only. At least ten valid trials and variation in all parameters are required for fANOVA."
        write_json(root / "importance.json", {"observations": len(complete), "importance": importance, "note": note})
        lines = [f"# {dataset} · joint CondCov Optuna", "", f"Stop reason {stop_reason}.",
                 f"Completed observations {len(complete)} ({new} new).",
                 f"IID target {DATASETS[dataset][1]:.1%}. The objective is mean source validation accuracy.", "",
                 "| Trial | Params | Source Val | IID | Test out |", "| --- | --- | ---: | ---: | ---: |"]
        for r in by_source:
            values = ", ".join(f"{key}={r['params'][key]}" for key in SPACE)
            lines.append(f"| {r['trial_id']} | {values} | {r['source_val']:.3%} | "
                         f"{r['iid']:.3%} | {r['test_out']:.3%} |")
        if source_best is not None:
            lines.extend(["", f"Source validation winner trial {source_best['trial_id']}.",
                          f"Target IID winner trial {iid_best['trial_id']}."])
        lines.extend(["", note])
        for key, score in importance.items():
            lines.append(f"- {key} {score:.4f}")
        write_text(root / "report.md", "\n".join(lines) + "\n")
    return status
