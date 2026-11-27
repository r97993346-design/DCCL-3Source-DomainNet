"""Cross-host Optuna registry backed by Git fast-forward updates."""
import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

BRANCH = "registry/global-optuna-trials"
REMOTE = "https://github.com/r97993346-design/DCCL-3Source-DomainNet.git"
REF = f"refs/heads/{BRANCH}"
DATASETS = {"PACS", "VLCS", "OfficeHome", "TerraIncognita"}
IDENTITY = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA = re.compile(r"^[0-9a-f]{64}$")
STATES = {"WAITING", "RUNNING", "COMPLETE", "FAILED", "PAUSED", "STALE"}


def canonical(value):
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite configuration value")
        return int(value) if value.is_integer() else value
    return value


def encode(value):
    return json.dumps(canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def configuration(dataset, dataset_version, weights_sha256, implementation_sha256, args, hparams, fold_seeds, schedule):
    if dataset not in DATASETS or not all(SHA.fullmatch(i) for i in (dataset_version, weights_sha256, implementation_sha256)):
        raise ValueError("dataset, weights and implementation identities are required")
    keys = ("algorithm", "steps", "checkpoint_freq", "batch_size", "source_sampling", "holdout_fraction",
            "trial_seed", "hparams_seed", "deterministic", "source_only_selection", "source_envs", "target_env",
            "fixed_fold_seed", "label_ratio", "evalmode", "aug", "cipt_cache_eval_visual")
    required = set(keys) - {"source_envs", "target_env", "fixed_fold_seed"}
    if required - args.keys() or len(fold_seeds) != 4 or len(set(fold_seeds)) != 4:
        raise ValueError("incomplete effective training protocol")
    params = {k: v for k, v in hparams.items() if k not in {"cipt_clip_path", "data_dir", "output_root", "cipt_class_names"}}
    if params["cipt_causal_contrastive_weight"] == 0:
        params["cipt_clip_teacher_mix"] = 0
    return canonical(dict(dataset=dataset, dataset_version=dataset_version, weights_sha256=weights_sha256,
                          implementation_sha256=implementation_sha256, args={k: args.get(k) for k in keys},
                          hparams=params, fold_seeds=fold_seeds, schedule=schedule,
                          objective="mean_four_fold_source_val", selection="earliest_max_source_val"))


def protocol_fingerprint(config):
    searchable = {"lr", "cipt_condcov_weight", "cipt_env_loss_weight", "cipt_causal_contrastive_weight",
                  "cipt_contrastive_weight", "cipt_clip_teacher_mix", "cipt_prompt_length"}
    return digest(dict(config, hparams={k: v for k, v in config["hparams"].items() if k not in searchable}))


def model_fingerprint(root):
    root = Path(root)
    # The entire imported training implementation and tokenizer data form the version boundary.
    files = sorted((*root.joinpath("domainbed").rglob("*.py"), root / "train_all.py",
                    *root.parents[1].joinpath("CLIP", "clip").glob("*.py"),
                    root.parents[1] / "CLIP/clip/bpe_simple_vocab_16e6.txt.gz"))
    fingerprints = []
    for path in files:
        code = path.read_bytes()
        if path.suffix == ".py":
            tree = ast.parse(code)
            if path.name == "train_all.py":
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "add_argument" and node.args
                            and isinstance(node.args[0], ast.Constant)
                            and node.args[0].value in {"--data_dir", "--output_root", "--cipt_clip_path"}):
                        for kw in node.keywords:
                            if kw.arg == "default":
                                kw.value = ast.Constant(None)
            code = ast.dump(tree).encode()
        fingerprints.append((str(path.relative_to(root.parents[1])), hashlib.sha256(code).hexdigest()))
    return digest(fingerprints)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Unavailable(RuntimeError):
    pass


class Registry:
    def __init__(self, remote=REMOTE):
        self.remote = remote

    def git(self, cwd, *args, allow_failure=False):
        import os
        try:
            result = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *args],
                                    cwd=cwd, capture_output=True, timeout=35,
                                    env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise Unavailable("Git transport unavailable") from exc
        if result.returncode and not allow_failure:
            raise Unavailable(f"Git {args[0]} failed ({result.returncode})")
        return result

    def checkout(self, directory, ref=REF):
        self.git(directory, "init", "-q")
        refs = self.git(directory, "ls-remote", "--refs", self.remote, ref).stdout.splitlines()
        if not refs:
            return False
        self.git(directory, "fetch", "-q", "--depth=1", self.remote, ref)
        self.git(directory, "checkout", "-q", "--detach", "FETCH_HEAD")
        return True

    def update(self, key, transform, *, tries=8):
        for attempt in range(tries):
            with tempfile.TemporaryDirectory() as directory:
                self.checkout(directory)
                path = Path(directory) / key
                previous = json.loads(path.read_bytes()) if path.exists() else None
                proposed = transform(previous)
                if proposed is None or proposed == previous:
                    return False, previous
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(encode(proposed) + b"\n")
                self.git(directory, "add", "--", str(key))
                self.git(directory, "-c", "user.name=Optuna registry", "-c", "user.email=optuna-registry@users.noreply.github.com",
                         "commit", "-qm", f"Update {key}")
                result = self.git(directory, "push", "--porcelain", self.remote, f"HEAD:{REF}", allow_failure=True)
                if result.returncode == 0:
                    return True, proposed
                # A rejected push or a lost acknowledgment never grants ownership.
                if b"non-fast-forward" not in result.stdout + result.stderr and b"fetch first" not in result.stdout + result.stderr and b"failed to update ref" not in result.stdout + result.stderr:
                    raise Unavailable("reservation update unconfirmed")
            time.sleep(min(.05 * 2**attempt, 1))
        raise Unavailable("registry contention did not resolve")

    @staticmethod
    def key(dataset, config_hash):
        if dataset not in DATASETS or not SHA.fullmatch(config_hash):
            raise ValueError("invalid registry key")
        return Path("trials", dataset, f"{config_hash}.json")

    def initialize(self):
        return self.update(Path("schema.json"), lambda old: old or {"version": 1, "type": "global-optuna-registry"})

    def records(self):
        with tempfile.TemporaryDirectory() as directory:
            self.checkout(directory)
            return [json.loads(path.read_bytes()) for path in sorted(Path(directory).glob("trials/*/*.json"))]

    def claim(self, config, *, server_id, study, trial_id, commit, owner_token, waiting=False):
        if not IDENTITY.fullmatch(server_id) or not owner_token:
            raise ValueError("explicit server identity and stable owner token required")
        key = self.key(config["dataset"], digest(config))
        timestamp = now()
        record = dict(config_hash=digest(config), dataset=config["dataset"], effective_config=config,
                      protocol_fingerprint=protocol_fingerprint(config), server_id=server_id,
                      local_study_name=study, local_trial_id=trial_id, status="WAITING" if waiting else "RUNNING",
                      owner_token=owner_token, created_at=timestamp, updated_at=timestamp, heartbeat_at=timestamp,
                      completed_folds=[], source_val=None, iid=None, test_out=None, result_branch=None,
                      result_path=None, training_code_commit=commit, error=None, retry_count=0)

        def reserve(old):
            if old is None:
                return record
            if not waiting and old["owner_token"] == owner_token and old["server_id"] == server_id:
                if old["status"] in {"WAITING", "PAUSED", "RUNNING"}:
                    return dict(old, status="RUNNING", updated_at=timestamp, heartbeat_at=timestamp)
            return None

        changed, row = self.update(key, reserve)
        return bool(not waiting and row and row["status"] == "RUNNING"
                    and row["owner_token"] == owner_token and row["server_id"] == server_id), row

    def progress(self, key, token, *, completed_folds=None, status="RUNNING", error=None):
        if status not in {"RUNNING", "PAUSED", "FAILED", "STALE"}:
            raise ValueError("invalid progress transition")

        def transition(old):
            if not old or old["owner_token"] != token or old["status"] != "RUNNING":
                return None
            folds = old["completed_folds"] if completed_folds is None else sorted(set(completed_folds))
            if not set(old["completed_folds"]) <= set(folds) <= set(range(4)):
                raise ValueError("fold progress must be monotonic")
            clock = now()
            return dict(old, status=status, completed_folds=folds, updated_at=clock, heartbeat_at=clock, error=error)

        return self.update(key, transition)

    def retry(self, key, *, previous_token, new_token, server_id, study, trial_id, commit):
        # Only the stopped owner may write FAILED/STALE via progress. Elapsed time is never proof.
        def recover(old):
            if not old or old["owner_token"] != previous_token or old["status"] not in {"FAILED", "STALE"}:
                return None
            clock = now()
            return dict(old, owner_token=new_token, server_id=server_id, local_study_name=study,
                        local_trial_id=trial_id, training_code_commit=commit, status="RUNNING",
                        completed_folds=[], error=None, retry_count=old["retry_count"] + 1,
                        heartbeat_at=clock, updated_at=clock)

        return self.update(key, recover)

    def published(self, branch, path, expected_hash):
        with tempfile.TemporaryDirectory() as directory:
            if not branch.startswith("results/") or Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("invalid result location")
            if not self.checkout(directory, f"refs/heads/{branch}"):
                raise Unavailable("result branch has not been published")
            root = Path(directory)
            manifest = json.loads((root / path).read_bytes())
            if manifest["config_hash"] != expected_hash or len(manifest["folds"]) != 4:
                raise ValueError("incompatible remote result")
            if [fold["fold"] for fold in manifest["folds"]] != list(range(4)):
                raise ValueError("four distinct folds required")
            for fold in manifest["folds"]:
                records = None
                for label in ("results", "learning_rates"):
                    entry = fold[label]
                    relative = Path(entry["path"])
                    if relative.is_absolute() or ".." in relative.parts or not SHA.fullmatch(entry["sha256"]):
                        raise ValueError("invalid result artifact reference")
                    artifact = root / relative
                    if not artifact.resolve().is_relative_to(root.resolve()) or hashlib.sha256(artifact.read_bytes()).hexdigest() != entry["sha256"]:
                        raise ValueError("result artifact does not match published digest")
                    if label == "results":
                        records = [json.loads(line) for line in artifact.read_bytes().splitlines() if line.strip()]
                selected = max(records, key=lambda row: row["train_out"])
                for key, expected in {"selected_step": selected["step"], "source_val": selected["train_out"],
                                      "iid": selected["test_in"], "test_out": selected["test_out"]}.items():
                    if fold[key] != expected:
                        raise ValueError(f"published fold summary disagrees with log: {key}")
            return manifest

    def complete(self, key, token, branch, path, verify):
        manifest = self.published(branch, path, key.stem)
        scores = verify(manifest)
        if set(scores) != {"source_val", "iid", "test_out"} or not all(0 <= val <= 1 for val in scores.values()):
            raise ValueError("invalid verified metrics")

        def finish(old):
            if not old or old["owner_token"] != token or old["status"] != "RUNNING":
                return None
            if old["completed_folds"] != list(range(4)):
                raise ValueError("cannot complete without four registered folds")
            return dict(old, status="COMPLETE", **scores, result_branch=branch, result_path=path, updated_at=now())

        return self.update(key, finish)


def import_observation(study, record, params, distributions, config, verified_metrics):
    import optuna

    if record["status"] != "COMPLETE" or record["completed_folds"] != list(range(4)):
        return False
    if record["config_hash"] != digest(config) or record["protocol_fingerprint"] != protocol_fingerprint(config):
        return False
    if record["effective_config"] != config or set(params) != set(distributions):
        return False
    if any(t.user_attrs.get("config_hash") == record["config_hash"] for t in study.get_trials(deepcopy=False)):
        return False
    if any(not math.isclose(record[k], verified_metrics[k], rel_tol=0, abs_tol=1e-12)
           for k in ("source_val", "iid", "test_out")):
        return False
    try:
        trial = optuna.trial.create_trial(params=params, distributions=distributions, value=verified_metrics["source_val"],
            user_attrs={"config_hash": record["config_hash"], "external": True, "server_id": record["server_id"],
                        "result_branch": record["result_branch"], "result_path": record["result_path"]})
    except ValueError:
        return False
    study.add_trial(trial)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init", "list"))
    parser.add_argument("--remote", default=REMOTE)
    args = parser.parse_args()
    registry = Registry(args.remote)
    if args.action == "init":
        print(registry.initialize())
    else:
        print(json.dumps(registry.records(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
