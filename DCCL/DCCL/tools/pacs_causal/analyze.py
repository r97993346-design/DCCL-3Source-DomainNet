"""Wait for the verified training folds, analyze true representations, and publish."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from experiment_archive import BRANCHES, REMOTE, git, safe
from tools.pacs_causal.queue import DOMAINS, GROUPS, PROJECT, ROOT, atomic_json, folds, say, wait_for_gpu


def done(group):
    return ROOT / "logs" / f"{group}.complete.json"


def wait_training(group):
    while not done(group).is_file():
        panes = subprocess.run(["tmux", "list-panes", "-t", f"=dicm_pacs_causal_{group}",
                                "-F", "#{pane_dead}"], text=True, capture_output=True)
        live = panes.returncode == 0 and "0" in panes.stdout.splitlines()
        if not live and not done(group).is_file():
            raise RuntimeError(f"{group} queue exited before completion. See logs/{group}.queue.log.")
        say("Waiting for completed training folds:", group)
        time.sleep(60)
    info = json.loads(done(group).read_text(encoding="utf-8"))
    assert info["folds"] == 4
    return info["commit"]


def recorded_code():
    commit = subprocess.check_output(["git", "-C", PROJECT, "rev-parse", "HEAD"],
                                     text=True).strip()
    if subprocess.check_output(["git", "-C", PROJECT, "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Run analysis from a clean code revision")
    return commit


def analyze_local(trained_commit, analysis_commit, gpu_uuid):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import torch
    from tools.pacs_causal.features import extract
    from tools.pacs_causal.metrics import analyze
    from tools.pacs_causal.report import reports, train_metrics, validation_pair
    from tools.pacs_causal.visualize import plots

    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    rows, diagnostics = [], []
    for index, domain in enumerate(DOMAINS):
        runs, folds_by_group, split_hash = validation_pair(index, trained_commit)
        rows.extend(train_metrics(index, g, folds_by_group[g], split_hash, trained_commit) for g in GROUPS)
        probe_dir = ROOT / "probes" / domain
        probe_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = probe_dir / "metrics.json"
        checkpoint = runs["P1"] / "checkpoints/iid_best.pth"
        with checkpoint.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else None
            if digest is None:
                digest_reader = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024**2), b""):
                    digest_reader.update(chunk)
                digest = digest_reader.hexdigest()
        features = ROOT / "features" / domain
        visuals = ROOT / "visualizations" / domain
        if metrics_file.is_file():
            result = json.loads(metrics_file.read_text(encoding="utf-8"))
            if (result["checkpoint_sha256"], result["analysis_commit"]) != (digest, analysis_commit):
                raise ValueError("Existing feature report used a different checkpoint or analysis code")
            assert (features / "features.npz").is_file()
            assert (visuals / "tsne_coordinates.csv").is_file()
        else:
            say("Extracting and scoring source representations for", domain, checkpoint)
            payload = extract(checkpoint, "cuda", features)
            assert payload["split_sha256"] == split_hash
            result = analyze(payload)
            result.update(held_out=domain, counts=payload["counts"], checkpoint=payload["checkpoint"],
                          checkpoint_sha256=digest, training_source=payload["provenance"],
                          analysis_commit=analysis_commit, split_sha256=split_hash)
            result["plot_settings"] = plots(payload, visuals)
            atomic_json(metrics_file, result)
            del payload
        diagnostics.append(result)
        say("Finished real feature analysis for", domain)
    reports(rows, diagnostics)
    return rows, diagnostics


def report_payload():
    result = {}
    for part, suffixes in (
        ("reports", {".md", ".csv", ".json"}),
        ("visualizations", {".png", ".pdf", ".json", ".csv"}),
        ("probes", {".json"}),
    ):
        for path in sorted((ROOT / part).rglob("*")):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if path.is_symlink() or path.stat().st_size > 25 * 1024**2:
                raise ValueError(f"Oversized or linked result {path.name}")
            data = (safe(path.read_text(encoding="utf-8")).encode()
                    if path.suffix in {".md", ".csv", ".json"} else path.read_bytes())
            result[path.relative_to(ROOT)] = data
    return result


def publish_report(trained_commit, analysis_commit):
    payload = report_payload()
    branch = "refs/heads/" + BRANCHES["DICM"]
    prefix = Path("pacs_causal_validation_20261119")
    manifest = dict(training_commit=trained_commit, analysis_commit=analysis_commit, folds={})
    for group in GROUPS:
        for index, domain in enumerate(DOMAINS):
            folder, = folds(group, index)
            run = json.loads((folder / ".github-archive/manifest.json").read_text())
            receipt = json.loads((folder / "archive_receipt.json").read_text())
            manifest["folds"][f"{group}_{domain}"] = dict(id=run["run_id"], **receipt)
    manifest["files"] = {str(k): hashlib.sha256(data).hexdigest() for k, data in payload.items()}
    payload[Path("manifest.json")] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.TemporaryDirectory(prefix="pacs-causal-publish-") as path:
        base = Path(path)
        git(base, "init", "-q")
        git(base, "fetch", "--depth=1", REMOTE, branch)
        git(base, "checkout", "--detach", "FETCH_HEAD")
        for row in manifest["folds"].values():
            if not (base / "runs" / row["id"] / "evaluation.json").is_file():
                raise RuntimeError("A recorded fold bundle is not visible on GitHub yet")
        if (base / prefix).exists():
            for rel, data in payload.items():
                if (base / prefix / rel).read_bytes() != data:
                    raise ValueError("GitHub already contains different content at the report path")
            return git(base, "rev-parse", "HEAD").decode().strip()
        for rel, data in payload.items():
            dst = base / prefix / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        git(base, "add", "--", str(prefix))
        git(base, "commit", "-m", "Archive PACS DICM paired validation and source-only feature diagnostics")
        commit = git(base, "rev-parse", "HEAD").decode().strip()
        git(base, "push", REMOTE, "HEAD:" + branch)
        ref = git(base, "ls-remote", "--exit-code", REMOTE, branch).decode().split()
        if not ref or ref[0] != commit:
            raise RuntimeError("Remote advanced during verification")
        return commit


def main():
    with (ROOT / "analysis.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        analysis_commit = recorded_code()
        trained = wait_training("P1")
        if wait_training("P0") != trained:
            raise ValueError("P0 and P1 training commits differ")
        uuid, status = wait_for_gpu(1)
        say("Feature extraction on GPU1", status)
        rows, diagnostics = analyze_local(trained, analysis_commit, uuid)
        atomic_json(ROOT / "reports/analysis_complete.json",
                    dict(training_commit=trained, analysis_commit=analysis_commit,
                         completed_folds=len(rows), analyzed_targets=len(diagnostics)))
        say("Chinese report, CSVs, PNG and PDF figures generated")
        while True:
            if not all((folder / "archive_receipt.json").is_file() for g in GROUPS
                       for i in range(len(DOMAINS)) for folder in folds(g, i)):
                say("Waiting for eight verified training archive receipts")
            else:
                try:
                    commit = publish_report(trained, analysis_commit)
                except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                    say("Final GitHub synchronization retry scheduled", type(error).__name__)
                else:
                    atomic_json(ROOT / "logs/final_archive_receipt.json", dict(commit=commit))
                    say("Final GitHub report verified", commit)
                    return
            time.sleep(60)


if __name__ == "__main__":
    main()
