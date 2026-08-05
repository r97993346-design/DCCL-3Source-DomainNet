#!/usr/bin/env python3
"""Target-SWAD staged hyperparameter search for the fixed PICCL v2 model.

Every search and confirmation stage maximizes the final target-domain ``SWAD``
mean. ``SWAD (inD)`` is retained only as a diagnostic value. This is an Oracle
target-domain selection protocol and is labelled as such in every summary.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean


SWAD_RE = re.compile(r"SWAD\s*=\s*([0-9]+(?:\.[0-9]+)?)%")
SWAD_IND_RE = re.compile(
    r"SWAD\s*\(inD\)\s*=\s*([0-9]+(?:\.[0-9]+)?)%",
)

# This parameter remains registered for compatibility, but the fixed v2
# forward path no longer consumes it. Searching it would create fake trials.
DEAD_SEARCH_PARAMS = {"piccl_residual_scale"}

# DCCL parameters stay fixed so that an improvement can be attributed to PICCL.
BASE_DCCL_SEARCH_PARAMS = {
    "batch_size",
    "l",
    "l_d",
    "lr",
    "t",
    "t_pre",
    "weight_decay",
}


def parse_swad_metrics(text: str) -> dict[str, float]:
    """Return the final target and source-validation SWAD values as fractions."""
    source_matches = SWAD_IND_RE.findall(text)
    target_matches = SWAD_RE.findall(text)
    if not target_matches:
        raise ValueError("log does not contain a final target-domain 'SWAD' result")
    result = {"swad_target": float(target_matches[-1]) / 100.0}
    if source_matches:
        result["swad_indomain_report_only"] = float(source_matches[-1]) / 100.0
    return result


def selection_objective(target_scores: list[float]) -> float:
    """Use the mean final target-domain SWAD as the Optuna objective."""
    if not target_scores:
        raise ValueError("at least one target-domain SWAD score is required")
    if any(not math.isfinite(value) for value in target_scores):
        raise ValueError("target-domain SWAD scores must be finite")
    return mean(target_scores)


def validate_search_space(search_space: dict) -> None:
    keys = set(search_space)
    dead = sorted(keys & DEAD_SEARCH_PARAMS)
    if dead:
        raise ValueError(f"dead v2 parameters cannot be searched: {dead}")
    base = sorted(keys & BASE_DCCL_SEARCH_PARAMS)
    if base:
        raise ValueError(f"DCCL baseline parameters must remain fixed: {base}")
    if {"piccl_warmup_ratio", "piccl_delayed_start_ratio"} <= keys:
        raise ValueError(
            "search only one of piccl_warmup_ratio and "
            "piccl_delayed_start_ratio; v2 uses their sum"
        )
    unsupported = sorted(key for key in keys if not key.startswith("piccl_"))
    if unsupported:
        raise ValueError(f"only PICCL-specific parameters may be searched: {unsupported}")


def suggest_params(trial, search_space: dict) -> dict:
    params = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif kind == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        else:
            raise ValueError(f"unsupported search type for {name}: {kind}")
    return params


def params_to_cli(params: dict) -> list[str]:
    result = []
    for key, value in sorted(params.items()):
        if isinstance(value, bool):
            value = str(value).lower()
        result.extend([f"--{key}", str(value)])
    return result


def build_command(
    cfg: dict,
    params: dict,
    task: dict,
    run_root: Path,
    data_dir: Path,
    steps: int,
    seed: int,
) -> list[str]:
    command = [
        sys.executable,
        str(cfg.get("train_script", "train_all.py")),
        f"{task['name']}_seed{seed}",
        "--dataset",
        str(cfg["dataset"]),
        "--algorithm",
        str(cfg.get("algorithm", "PICCL")),
        "--model",
        str(cfg.get("model", "resnet50")),
        "--data_dir",
        str(data_dir),
        "--output_root",
        str(run_root),
        "--steps",
        str(steps),
        "--checkpoint_freq",
        str(cfg.get("checkpoint_freq", 100)),
        "--seed",
        str(seed),
        "--trial_seed",
        str(cfg.get("trial_seed", 0)),
    ]
    if cfg.get("deterministic", True):
        command.append("--deterministic")
    if "test_envs" in task:
        command.extend(["--test_envs", *[str(x) for x in task["test_envs"]]])
    if "source_envs" in task:
        command.extend(["--source_envs", *[str(x) for x in task["source_envs"]]])
    if "target_env" in task:
        command.extend(["--target_env", str(task["target_env"])])
    command.extend([str(x) for x in cfg.get("base_args", [])])
    command.extend(params_to_cli(params))
    return command


def latest_log(run_root: Path) -> Path | None:
    candidates = list(run_root.glob("*/log.txt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_task(
    cfg: dict,
    params: dict,
    task: dict,
    task_root: Path,
    data_dir: Path,
    repo_root: Path,
    steps: int,
    seed: int,
    gpu: str | None,
    resume: bool,
) -> dict:
    metrics_path = task_root / "metrics.json"
    if resume and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    task_root.mkdir(parents=True, exist_ok=True)
    command = build_command(cfg, params, task, task_root, data_dir, steps, seed)
    (task_root / "command.json").write_text(
        json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    process = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )
    (task_root / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (task_root / "stderr.log").write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"{task['name']} exited with code {process.returncode}")

    log_path = latest_log(task_root)
    log_text = log_path.read_text(errors="ignore") if log_path else process.stdout
    metrics = {
        "task": task["name"],
        "seed": seed,
        "steps": steps,
        "git_commit": git_commit(repo_root),
        **parse_swad_metrics(log_text),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics


def evaluate_params(
    cfg: dict,
    params: dict,
    trial_root: Path,
    data_dir: Path,
    repo_root: Path,
    steps: int,
    seed: int,
    gpus: list[str],
    max_concurrent: int,
    resume: bool,
) -> dict:
    trial_root.mkdir(parents=True, exist_ok=True)
    (trial_root / "params.json").write_text(
        json.dumps(params, indent=2, sort_keys=True), encoding="utf-8"
    )
    tasks = cfg["tasks"]
    workers = max(1, min(max_concurrent, len(tasks)))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for index, task in enumerate(tasks):
            gpu = gpus[index % len(gpus)] if gpus else None
            futures.append(
                pool.submit(
                    run_task,
                    cfg,
                    params,
                    task,
                    trial_root / task["name"],
                    data_dir,
                    repo_root,
                    steps,
                    seed,
                    gpu,
                    resume,
                )
            )
        for future in futures:
            results.append(future.result())

    target_scores = [item["swad_target"] for item in results]
    objective = selection_objective(target_scores)
    summary = {
        "objective": objective,
        "selection_protocol": "target_swad_oracle",
        "selection_metric": "mean_target_swad",
        "target_swad": target_scores,
        "source_swad_indomain_report_only": [
            item.get("swad_indomain_report_only") for item in results
        ],
        "task_metrics": results,
    }
    (trial_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def inherited_params(cfg: dict, output_root: Path, stage_name: str) -> dict:
    stage = cfg["stages"][stage_name]
    params = dict(cfg.get("fixed_params", {}))
    parent = stage.get("inherits")
    if not parent:
        return params
    candidates = [
        output_root / parent / "confirmation" / "best_config.json",
        output_root / parent / "best_config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            params.update(json.loads(candidate.read_text(encoding="utf-8")))
            return params
    raise FileNotFoundError(
        f"stage '{stage_name}' requires the completed '{parent}' stage"
    )


def ranked_trials(study, base_params: dict) -> list[dict]:
    rows = []
    for trial in study.trials:
        if trial.value is None or trial.state.name != "COMPLETE":
            continue
        rows.append(
            {
                "trial": trial.number,
                "objective": float(trial.value),
                "selection_protocol": "target_swad_oracle",
                "selection_metric": "mean_target_swad",
                "params": {**base_params, **trial.params},
            }
        )
    return sorted(rows, key=lambda row: row["objective"], reverse=True)


def run_search(args, cfg: dict, repo_root: Path, output_root: Path) -> None:
    try:
        import optuna
    except ImportError as error:
        raise SystemExit("Optuna is required: pip install optuna") from error

    stage_cfg = cfg["stages"][args.stage]
    search_space = stage_cfg["search_space"]
    validate_search_space(search_space)
    base_params = inherited_params(cfg, output_root, args.stage)
    stage_root = output_root / args.stage
    stage_root.mkdir(parents=True, exist_ok=True)

    sampler_cfg = cfg.get("optuna", {})
    sampler = optuna.samplers.TPESampler(
        seed=int(sampler_cfg.get("seed", 0)),
        n_startup_trials=int(sampler_cfg.get("n_startup_trials", 8)),
        multivariate=bool(sampler_cfg.get("multivariate", True)),
        group=bool(sampler_cfg.get("group", True)),
    )
    study = optuna.create_study(
        study_name=f"{cfg['study_name']}_{args.stage}",
        direction="maximize",
        sampler=sampler,
        storage=f"sqlite:///{stage_root / 'study.db'}",
        load_if_exists=True,
    )
    steps = int(args.steps or stage_cfg["steps"])

    def objective(trial):
        params = {**base_params, **suggest_params(trial, search_space)}
        summary = evaluate_params(
            cfg,
            params,
            stage_root / "trials" / f"trial_{trial.number:04d}",
            args.data_dir,
            repo_root,
            steps,
            args.seed,
            args.gpus,
            args.max_concurrent,
            args.resume,
        )
        trial.set_user_attr("target_swad", summary["target_swad"])
        trial.set_user_attr(
            "source_swad_indomain_report_only",
            summary["source_swad_indomain_report_only"],
        )
        return summary["objective"]

    n_trials = int(args.n_trials or stage_cfg["n_trials"])
    remaining_trials = max(n_trials - len(study.trials), 0)
    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials)
    ranking = ranked_trials(study, base_params)
    (stage_root / "ranking.json").write_text(
        json.dumps(ranking, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(
        stage_root / "ranking.csv",
        [
            {
                "trial": row["trial"],
                "objective": row["objective"],
                "selection_protocol": row["selection_protocol"],
                "selection_metric": row["selection_metric"],
                **row["params"],
            }
            for row in ranking
        ],
    )
    if ranking:
        (stage_root / "best_config.json").write_text(
            json.dumps(ranking[0]["params"], indent=2, sort_keys=True),
            encoding="utf-8",
        )


def run_confirmation(args, cfg: dict, repo_root: Path, output_root: Path) -> None:
    stage_root = output_root / args.stage
    ranking_path = stage_root / "ranking.json"
    if not ranking_path.exists():
        raise FileNotFoundError(f"run the '{args.stage}' search before confirmation")
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))[: args.top_k]
    full_steps = int(args.full_steps or cfg["full_steps"])
    confirmation_root = stage_root / "confirmation"
    rows = []
    for candidate_index, candidate in enumerate(ranking):
        target_scores = []
        source_indomain_report_only = []
        for seed in args.confirm_seeds:
            summary = evaluate_params(
                cfg,
                candidate["params"],
                confirmation_root / f"candidate_{candidate_index:02d}" / f"seed_{seed}",
                args.data_dir,
                repo_root,
                full_steps,
                seed,
                args.gpus,
                args.max_concurrent,
                args.resume,
            )
            target_scores.extend(summary["target_swad"])
            source_indomain_report_only.extend(
                summary["source_swad_indomain_report_only"]
            )
        rows.append(
            {
                "candidate": candidate_index,
                "search_trial": candidate["trial"],
                "objective": selection_objective(target_scores),
                "selection_protocol": "target_swad_oracle",
                "selection_metric": "mean_target_swad",
                "target_swad": target_scores,
                "source_swad_indomain_report_only": source_indomain_report_only,
                "params": candidate["params"],
            }
        )
    rows.sort(key=lambda row: row["objective"], reverse=True)
    confirmation_root.mkdir(parents=True, exist_ok=True)
    (confirmation_root / "ranking.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(
        confirmation_root / "ranking.csv",
        [
            {
                "candidate": row["candidate"],
                "search_trial": row["search_trial"],
                "objective": row["objective"],
                "selection_protocol": row["selection_protocol"],
                "selection_metric": row["selection_metric"],
                **row["params"],
            }
            for row in rows
        ],
    )
    if rows:
        (confirmation_root / "best_config.json").write_text(
            json.dumps(rows[0]["params"], indent=2, sort_keys=True),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--stage", choices=["core", "schedule"], default="core")
    parser.add_argument("--mode", choices=["search", "confirm"], default="search")
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--full-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--confirm-seeds", default="0,1,2")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.gpus = [item for item in args.gpus.split(",") if item != ""]
    args.confirm_seeds = [
        int(item) for item in args.confirm_seeds.split(",") if item != ""
    ]
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.stage not in cfg["stages"]:
        raise KeyError(f"stage '{args.stage}' is not defined in {args.config}")
    repo_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    data_dir = args.data_dir.resolve()
    args.data_dir = data_dir
    if args.mode == "search":
        run_search(args, cfg, repo_root, output_root)
    else:
        run_confirmation(args, cfg, repo_root, output_root)


if __name__ == "__main__":
    main()
