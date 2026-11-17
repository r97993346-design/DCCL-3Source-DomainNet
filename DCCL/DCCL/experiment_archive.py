"""Archive completed DomainBed runs in the project's private GitHub results branch."""
import csv
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

REMOTE = "git@github.com:r97993346-design/DCCL-DomainNet-Improve.git"
BRANCHES = {"CIPT": "experiments/cipt", "DICM": "experiments/dicm"}
METRICS = ("test_in", "test_out", "train_out")
FILES = ("manifest.json", "log.txt", "results.jsonl", "evaluation.json")
FIELDS = ("run_id", "completed_utc", "dataset", "algorithm", "seed", "code_branch",
          "code_commit", "fingerprint", "duplicate_of", "target", "IID", "SWAD", *METRICS)
SECRET = re.compile(
    r"(?i)github_pat_|gh[pousr]_[a-z0-9]+|AKIA[A-Z0-9]{16}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer\s+\S+|"
    r"https?://[^/\s]+@|(?:password|passwd|api[-_ ]?key|access[-_ ]?token|"
    r"secret[-_ ]?key|authorization|credential)[\"']?\s*(?:[:=]|\s)\s*\S+"
)
PATH = re.compile(r'(?<![\w/])(?:/(?:home|root|mnt|data|datasets|tmp|media|srv|opt|var|workspace|Users|autodl[^/]*)/|[A-Za-z]:[\\/])[^\s"\x27,;\])]+')
RUNTIME_ARGS = {"name", "unique_name", "out_dir", "out_root", "data_dir", "gpu",
                "device", "model_save", "prebuild_cache", "num_workers"}

_SWAD_CAPTURE = ContextVar("experiment_swad_capture", default=None)


def record_swad(results, start, end, count, **status):
    capture = _SWAD_CAPTURE.get()
    if capture is not None:
        capture.update(metrics=dict(results), start=start, end=end, count=count, **status)


def git(directory, *args):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0",
               GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=12")
    return subprocess.run(["git", "-C", str(directory), *args], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True, timeout=90).stdout


def safe(text):
    if SECRET.search(text):
        raise ValueError("possible credential in archive materials")
    return PATH.sub("<local-path>", text)


def json_text(value):
    def encode(item):
        if hasattr(item, "item"):
            return item.item()
        return str(item)

    return json.dumps(value, default=encode, ensure_ascii=False,
                      sort_keys=True, indent=2) + "\n"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, text):
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)


class Experiment:
    def __init__(self, project, args, dataset, logger):
        self.project = project
        self.args = args
        self.dataset = dataset
        self.logger = logger
        self.folds = []
        self.provenance = None
        if not args.debug:
            try:
                cwd = Path(__file__).parent
                code = git(cwd, "rev-parse", "HEAD").decode().strip()
                branch = git(cwd, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
                modified = git(cwd, "diff", "HEAD", "--", "*.py", "*.yaml", "*.yml")
                untracked = git(cwd, "ls-files", "-z", "--others", "--exclude-standard",
                                "--", "*.py", "*.yaml", "*.yml").split(b"\0")
                code_hash = hashlib.sha256(modified)
                for path in sorted(filter(None, untracked)):
                    # Ignore run outputs and generated configuration in the source digest.
                    if not set(os.fsdecode(path).split("/")) & {"train_output", "logs"}:
                        code_hash.update(path + b"\0" + (cwd / os.fsdecode(path)).read_bytes())
                self.provenance = {"commit": code, "branch": branch, "changes": code_hash.hexdigest()}
            except (OSError, subprocess.SubprocessError) as error:
                logger.warning("Could not read Git provenance (%s)", type(error).__name__)

    def wrap(self, train):
        def run(test_envs, *positional, **kwargs):
            args = kwargs["args"]
            parameters = dict(vars(args))
            hparams = dict(kwargs["hparams"])
            swad = {}
            token = _SWAD_CAPTURE.set(swad)
            try:
                selection, records = train(test_envs, *positional, **kwargs)
            finally:
                _SWAD_CAPTURE.reset(token)
            if not self.args.debug:
                target_env = kwargs.get("target_env")
                if target_env is None:
                    target_env = getattr(args, "target_env", None)
                targets = [target_env] if target_env is not None else test_envs
                iid = max(records, key=lambda row: row["train_out"])
                self.folds.append({
                    "target": [str(self.dataset.environments[index]) for index in targets],
                    "args": parameters, "hparams": hparams,
                    "steps": kwargs["n_steps"], "checkpoint_freq": kwargs["checkpoint_freq"],
                    "selection": dict(selection), "swad_evaluation": swad or None,
                    "iid_checkpoint": {key: iid.get(key) for key in ("step", *METRICS)},
                    "checkpoints": list(records),
                })
            return selection, records
        return run

    def finish(self):
        if self.args.debug:
            return
        try:
            if not self.provenance or not self.folds:
                raise ValueError("missing code provenance or evaluation")
            directory = self.args.out_dir
            run_id = uuid.uuid4().hex
            configs = [{
                "args": {k: v for k, v in fold["args"].items() if k not in RUNTIME_ARGS},
                "hparams": fold["hparams"], "steps": fold["steps"],
                "checkpoint_freq": fold["checkpoint_freq"], "target": fold["target"],
            } for fold in self.folds]
            fingerprint = digest(safe(json_text([
                self.provenance["commit"], self.provenance["changes"], configs
            ])).encode())
            manifest = {
                "run_id": run_id, "project": self.project,
                "evaluation_protocol": ("oracle-guided HPO" if
                                        getattr(self.args, "oracle_guided_hpo", False) else None),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": self.args.dataset, "algorithm": self.args.algorithm,
                "seed": getattr(self.args, "seed", None), "code": self.provenance,
                "fingerprint": fingerprint,
                "command": shlex.join([sys.executable, *sys.argv]),
                "checkpoint_files": [
                    path.name for path in sorted((directory / "checkpoints").glob("*.pth"))
                ],
            }
            payload = {
                "manifest.json": json_text(manifest),
                "evaluation.json": json_text(self.folds),
                **{name: (directory / name).read_text(encoding="utf-8")
                   for name in ("log.txt", "results.jsonl")},
            }
            outbox = directory / ".github-archive"
            sanitized = {name: safe(contents) for name, contents in payload.items()}
            with tempfile.TemporaryDirectory(prefix=".github-archive-", dir=directory) as temporary:
                staged = Path(temporary) / "archive"
                staged.mkdir()
                for name, contents in sanitized.items():
                    write_new(staged / name, contents)
                staged.rename(outbox)
            self.logger.info("GitHub archive verified: %s", publish(outbox))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.logger.warning("GitHub archive incomplete (%s). Check %s",
                                type(error).__name__, self.args.out_dir / ".github-archive")


def history_row(manifest, fold, duplicate_of):
    selection = fold["selection"]
    return {
        **{key: manifest[key] for key in FIELDS[:5]},
        "algorithm": fold["args"].get("algorithm", manifest["algorithm"]),
        "code_branch": manifest["code"]["branch"],
        "code_commit": manifest["code"]["commit"],
        "fingerprint": manifest["fingerprint"],
        "target": ",".join(fold["target"]), "duplicate_of": duplicate_of,
        "IID": selection.get("iid"),
        "SWAD": (selection.get("SWAD") if
                 (fold.get("swad_evaluation") or {}).get("converged") is not False else ""),
        **{key: fold["iid_checkpoint"][key] for key in METRICS},
    }


def publish(outbox, remote=REMOTE):
    payload = {}
    for name in FILES:
        path = outbox / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size >= 95_000_000:
            raise ValueError("missing, linked or oversized archive material")
        payload[name] = safe(path.read_text(encoding="utf-8"))
    manifest = json.loads(payload["manifest.json"])
    if not re.fullmatch("[0-9a-f]{32}", manifest["run_id"]):
        raise ValueError("invalid run ID")
    if manifest.get("project") not in BRANCHES:
        raise ValueError("unknown project")
    branch = "refs/heads/" + BRANCHES[manifest["project"]]
    relative_run = Path("runs") / manifest["run_id"]
    folds = json.loads(payload["evaluation.json"])
    for _attempt in range(3):
        with tempfile.TemporaryDirectory(prefix="experiment-github-") as temporary:
            checkout = Path(temporary)
            git(checkout, "init", "-q")
            git(checkout, "fetch", "--depth=1", remote, branch)
            base = git(checkout, "rev-parse", "FETCH_HEAD").decode().strip()
            git(checkout, "checkout", "--detach", base)
            destination = checkout / relative_run
            if destination.exists():
                if any((destination / name).read_text(encoding="utf-8") != contents
                       for name, contents in payload.items()):
                    raise ValueError("run ID collision")
                return base  # Already verified by fetching the remote branch.
            destination.mkdir(parents=True)
            for name, contents in payload.items():
                write_new(destination / name, contents)
            history = checkout / "history.csv"
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, FIELDS, lineterminator="\n")
            duplicate_of = ""
            if history.exists():
                with history.open(encoding="utf-8", newline="") as stream:
                    duplicate_of = next(
                        (row["run_id"] for row in csv.DictReader(stream)
                         if row["fingerprint"] == manifest["fingerprint"]), ""
                    )
            else:
                writer.writeheader()
            writer.writerows(history_row(manifest, fold, duplicate_of) for fold in folds)
            with history.open("a", encoding="utf-8", newline="") as stream:
                stream.write(buffer.getvalue())
            git(checkout, "add", "--", str(relative_run), "history.csv")
            git(checkout, "commit", "-qm", f"Archive {manifest['project']} {manifest['run_id']}")
            commit = git(checkout, "rev-parse", "HEAD").decode().strip()
            try:
                git(checkout, "push", remote, "HEAD:" + branch)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # A racing update or a lost acknowledgement is reconciled by fetching again.
                continue
            ref = git(checkout, "ls-remote", "--exit-code", remote, branch).decode().split()
            if ref and ref[0] == commit:
                return commit
    raise ValueError("GitHub archive could not be verified after three attempts")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python experiment_archive.py /path/to/run/.github-archive")
    print("GitHub archive verified:", publish(Path(sys.argv[1])))
