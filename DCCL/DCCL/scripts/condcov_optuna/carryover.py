"""Reuse compatible completed folds from the previous contrastive search."""

from protocol import (
    BASE, DATASETS, OUTPUT, SEARCH_KEYS, SPACE, hparams_for, read_json, require,
    signature, summarize,
)

LEGACY_NAME = "condcov_optuna_joint_20261125"
LEGACY_CODE = BASE.with_name(LEGACY_NAME) / "code" / "DCCL" / "DCCL"
LEGACY_OUTPUT = LEGACY_CODE / "train_output" / LEGACY_NAME


def check_protocol():
    old_plan = read_json(LEGACY_OUTPUT / "plan.json")
    new_plan = read_json(OUTPUT / "plan.json")
    old_plan.pop("code_commit")
    new_plan.pop("code_commit")
    old_choices = old_plan["space"].pop("contrastive_weight")
    new_choices = new_plan["space"].pop("contrastive_weight")
    require(old_choices == [0.1, 0.25, 0.5] and new_choices == [0.025, 0.05, 0.1]
            and old_plan == new_plan, "prior training or search protocol is incompatible")


def completed_attempt(trial, dataset, fold, params, read_attempt):
    for attempt in sorted((trial / f"env{fold}").glob("attempt_*"), reverse=True):
        if (attempt / "exit_code").is_file():
            try:
                return read_attempt(attempt, dataset, fold, params)
            except (ValueError, KeyError, OSError):
                continue
    return None


def collect(read_attempt):
    check_protocol()
    previous = {dataset: {"imports": [], "pending": [], "folds": {}} for dataset in DATASETS}
    for dataset, entries in previous.items():
        trial_configs = (LEGACY_OUTPUT / dataset / "trials").glob("trial_*/config.json")
        for config in sorted(trial_configs):
            hparams = read_json(config)
            params = {name: hparams.get(key) for name, key in SEARCH_KEYS.items()}
            if any(value not in SPACE[name] for name, value in params.items()):
                continue
            require(hparams == hparams_for(params), f"prior trial configuration mismatch in {config}")
            vector = signature(params)
            folds = entries["folds"].setdefault(vector, {})
            for fold in range(len(DATASETS[dataset][0])):
                if fold not in folds:
                    result = completed_attempt(config.parent, dataset, fold, params, read_attempt)
                    if result is not None:
                        folds[fold] = result
        for vector, folds in entries["folds"].items():
            params = dict(zip(SPACE, vector))
            if len(folds) == len(DATASETS[dataset][0]):
                entries["imports"].append({
                    **summarize([folds[fold] for fold in sorted(folds)]),
                    "params": params, "stop_reason": "reused_previous_search",
                })
            elif folds:
                entries["pending"].append(params)
    return previous
