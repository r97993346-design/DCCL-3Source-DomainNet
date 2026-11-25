"""Find compatible CondCov observations and reference-only configurations."""

import collections
import itertools
import json
import os
import re
import subprocess
from pathlib import Path

from protocol import (
    BASE, CODE, DATASETS, MODEL_COMMIT, OUTPUT, SEARCH_KEYS, git, read_json, read_rows,
    require, signature, summarize, validate_fold,
)


GRID_SPECS = (
    ("pacs_condcov_grid_search_20261124", "pacs_condcov_grid_20261124", 36),
    ("vlcs_condcov_s_grid_20261124", "vlcs_condcov_s_grid_20261124", 48),
    ("oh_condcov_grid_stage1_20261124", "oh_condcov_grid_stage1_20261124", 36),
)
DATA_VOLUME = BASE.parent
PROJECTS = Path("/home/hooasia/lgg/projects")
HISTORY_ROOTS = (PROJECTS, DATA_VOLUME / "lgg")
CODE_PATHS = ("train_all.py", "domainbed", "config.yaml")
SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules", "checkpoints", "weights",
        "datasets", "dataset", "data", "wandb", "runs", "preflight"}


def output_for_grid(folder):
    return DATA_VOLUME / folder / "code" / "DCCL" / "DCCL" / "train_output" / folder


def history_files():
    roots = list(HISTORY_ROOTS)
    roots.extend(path for path in DATA_VOLUME.iterdir()
                 if path.is_dir() and path != BASE and re.search(r"condcov|cipt|optuna", path.name, re.I))
    seen = set()
    for root in roots:
        for directory, subdirs, files in os.walk(root, followlinks=False):
            subdirs[:] = sorted(subdir for subdir in subdirs if subdir not in SKIP and not subdir.startswith("."))
            if "results.jsonl" in files:
                path = (Path(directory) / "results.jsonl").resolve()
                if path not in seen and OUTPUT not in path.parents:
                    seen.add(path)
                    yield path


def provenance(path):
    sha = None
    repo = None
    for parent in path.parents:
        if sha is None:
            plan = parent / "plan.json"
            stamp = parent / "code_commit"
            if plan.is_file():
                try:
                    candidate = read_json(plan).get("code_commit")
                except (ValueError, OSError, AttributeError):
                    candidate = None
                if isinstance(candidate, str) and re.fullmatch("[0-9a-f]{40}", candidate):
                    sha = candidate
            if sha is None and stamp.is_file():
                candidate = stamp.read_text(encoding="utf-8").strip()
                if re.fullmatch("[0-9a-f]{40}", candidate):
                    sha = candidate
        if repo is None and (parent / ".git").exists():
            repo = parent / "DCCL" / "DCCL"
        if repo is None and (parent / "code" / ".git").exists():
            repo = parent / "code" / "DCCL" / "DCCL"
        if sha is not None and repo is not None:
            return sha, repo
    return sha, repo


def model_matches(sha, repo):
    if not sha:
        return False
    if sha == MODEL_COMMIT:
        return True
    try:
        # Match the model, losses, runtime code, datasets, and evaluation protocol.
        return not git("diff", "--name-only", sha, MODEL_COMMIT, "--", *CODE_PATHS, cwd=repo or CODE)
    except subprocess.CalledProcessError:
        return False


def select_folds(rows, dataset, params, path):
    groups = collections.defaultdict(list)
    for row in rows:
        args = row["args"]
        if args["dataset"] == dataset and args["real_test_envs"] in ([0], [1], [2], [3]):
            groups[args["real_test_envs"][0]].append(row)
    folds = []
    for fold, checkpoints in groups.items():
        result = validate_fold(checkpoints, dataset, fold, params, imported=True)
        result["log"] = str(path)
        elapsed = path.parents[3] / "elapsed_seconds"
        if len(groups) == 1 and elapsed.is_file():
            result["elapsed_seconds"] = float(elapsed.read_text())
        folds.append(result)
    return folds


def collect():
    compatible = collections.defaultdict(dict)
    reference_groups = collections.defaultdict(dict)
    counts = {ds: collections.Counter() for ds in DATASETS}
    checked_code = {}
    for path in history_files():
        try:
            with path.open(encoding="utf-8") as stream:
                first = json.loads(next(stream))
            dataset = first.get("args", {}).get("dataset")
            if dataset not in DATASETS or first.get("ind_mode") != "condcov":
                continue
            hp = first["hparams"]
            params = {key: hp[config_key] for key, config_key in SEARCH_KEYS.items()}
            vector = signature(params)
            rows = read_rows(path)
            sha, repo = provenance(path)
            # Aggregate by code revision plus the logged configuration, never by target metrics.
            stable = {key: value for key, value in hp.items() if key != "ind_grad_steps"}
            identifier = (dataset, vector, sha, json.dumps(stable, sort_keys=True))
            if hp.get("lr") != 1e-5:
                counts[dataset]["other_learning_rate"] += 1
                if hp.get("cipt_s_uniform_weight") == 1 and hp.get("cipt_e_cls_weight") == 1:
                    for fold, group in itertools.groupby(rows, lambda r: tuple(r["args"].get("real_test_envs", ()))):
                        group = list(group)
                        if len(fold) == 1 and fold[0] in range(4) and group[-1]["step"] in (4999, 5000):
                            selection = max(row["train_out"] for row in group)
                            reference_groups[identifier].setdefault(fold[0], selection)
                continue
            if (sha, repo) not in checked_code:
                checked_code[sha, repo] = model_matches(sha, repo)
            require(checked_code[sha, repo], "model code provenance unverified")
            for fold in select_folds(rows, dataset, params, path):
                compatible[identifier].setdefault(fold["fold"], fold)
            counts[dataset]["compatible_files"] += 1
        except (KeyError, TypeError, ValueError, StopIteration, OSError) as error:
            if locals().get("dataset") in counts:
                counts[dataset][f"excluded: {error}"] += 1
        finally:
            dataset = None
    observations = {ds: {} for ds in DATASETS}
    for (dataset, vector, sha, _), folds in compatible.items():
        if sorted(folds) == list(range(4)):
            item = {"params": dict(zip(SEARCH_KEYS, vector)), "code_commit": sha,
                    **summarize([folds[i] for i in range(4)])}
            observations[dataset].setdefault(vector, item)
    references = {}
    for ds in DATASETS:
        possible = [(sum(folds.values()) / 4, vector)
                    for (dataset, vector, _, _), folds in reference_groups.items()
                    if dataset == ds and sorted(folds) == list(range(4))]
        if possible:
            source, vector = max(possible, key=lambda item: item[0])
            references[ds] = {"params": dict(zip(SEARCH_KEYS, vector)), "reference_source_val": source,
                              "use": "initial_candidate_only"}
    manifest = {ds: {"imports": list(observations[ds].values()), "reference": references.get(ds),
                     "history_counts": dict(counts[ds])} for ds in DATASETS}
    return manifest
