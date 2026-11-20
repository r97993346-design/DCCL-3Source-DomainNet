"""Append verified DirectDiff16 bundles to the existing DICM GitHub archive."""

import fcntl
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time

from tools.pacs_directdiff.common import DOMAINS, GROUPS, ROOT, atomic_json, folds, say
from tools.pacs_directdiff.queue import completed
from tools.pacs_projection.archive import publish, safe_file


PREFIX = Path("pacs_directdiff16_20261120")


def fold_payload(folder):
    names = ("log.txt", "manifest.json", "evaluation.json", "results.jsonl")
    payload = {Path(name): safe_file(folder / ".github-archive" / name) for name in names}
    identity = sha256((folder / "sample_manifest.json").read_bytes()).hexdigest()
    payload[Path("sample_manifest_sha256.txt")] = (identity + "\n").encode()
    for event in sorted((folder / "gate_updates").glob("*.json")):
        payload[event.relative_to(folder)] = safe_file(event)
    final_gate = folder / "swad_gate.json"
    if final_gate.is_file():
        payload[Path(final_gate.name)] = safe_file(final_gate)
    return payload


def report_payload():
    files = {}
    for name in ("reports", "probes", "visualizations", "offline", "configs"):
        for path in sorted((ROOT / name).rglob("*")):
            if path.suffix in {".csv", ".json", ".md", ".png", ".pdf", ".txt"}:
                files[path.relative_to(ROOT)] = safe_file(path)
    for log in ("offline-gate", "config-check", "cpu-check", "Uniform-cuda-check", "Gate-cuda-check",
                "postprocess-check"):
        p = ROOT / "logs" / f"{log}.log"
        files[p.relative_to(ROOT)] = safe_file(p)
    for p in sorted((ROOT / "logs").glob("*.json")):
        if "failure" in p.name or "complete" in p.name or "passed" in p.name:
            files[p.relative_to(ROOT)] = safe_file(p)
    return files


def run():
    with (ROOT / "archive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        commit = json.loads((ROOT / "logs" / "preflight.passed.json").read_text())["commit"]
        marker = ROOT / "logs" / "report.complete.json"
        while True:
            count = 0
            for group in GROUPS:
                for index, domain in enumerate(DOMAINS):
                    candidates = folds(group, index)
                    if not candidates or not (candidates[0] / ".github-archive").is_dir():
                        continue
                    folder, = candidates
                    receipt = folder / "archive_receipt.json"
                    if not receipt.is_file():
                        completed(folder, group, index, commit)
                        destination = PREFIX / "folds" / group / domain
                        try:
                            revision = publish(destination, fold_payload(folder))
                        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                            say("Fold upload retry", group, domain, type(error).__name__)
                            continue
                        atomic_json(receipt, dict(commit=revision, archive_path=str(destination)))
                        say("GitHub fold verified", group, domain, revision)
                    count += 1
            if marker.is_file() and count == json.loads(marker.read_text())["verified_training_folds"]:
                destination = PREFIX / "analysis"
                try:
                    revision = publish(destination, report_payload())
                except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                    say("Report upload retry", type(error).__name__)
                else:
                    atomic_json(ROOT / "logs" / "final_archive_receipt.json",
                                dict(commit=revision, archive_path=str(destination)))
                    say("GitHub report verified", revision)
                    return
            time.sleep(60)


if __name__ == "__main__":
    run()
