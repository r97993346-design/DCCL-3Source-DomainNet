#!/usr/bin/env python3
"""Evaluate saved DomainBed CIPT and upstream CIPT on one shared image manifest.

No optimization, label-conditioned inference, alpha selection, or checkpoint
selection is performed. Each model retains its own learned prompts, frozen CLIP
precision, and native predict() implementation. Source training images use eval
preprocessing. Run from the training repository's DCCL/DCCL directory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gc
import hashlib
import io
import inspect
import json
import math
from pathlib import Path, PurePosixPath
import random
import subprocess
import sys
import tarfile
import tempfile

UPSTREAM_COMMIT = "a805d878acc7d79778d1ec1c1e4d73ba6aff334b"
AUDIT_REVISION = "2026-10-31-pinned-source-resume"
PARTITIONS = ("source_train", "source_val", "target_in", "target_out")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(root, *args):
    try:
        return subprocess.check_output(["git", "-C", str(root), *args],
                                       stderr=subprocess.DEVNULL, text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        return None


@contextmanager
def clean_upstream_snapshot(root, commit=UPSTREAM_COMMIT):
    """Read the pinned Git objects into a private copy; retain the working tree."""
    upstream = Path(root) / "third_party" / "CIPT"
    toplevel = git_output(upstream, "rev-parse", "--show-toplevel")
    if toplevel is None or Path(toplevel).resolve() != upstream.resolve():
        raise ValueError(f"Initialize the CIPT submodule first: {upstream}")
    try:
        payload = subprocess.check_output(
            ["git", "-C", str(upstream), "archive", "--format=tar", commit, "--", "cipt"],
            stderr=subprocess.PIPE, timeout=30)
        diff = subprocess.check_output(
            ["git", "-C", str(upstream), "diff", "--no-ext-diff", "--no-textconv", "HEAD", "--"],
            stderr=subprocess.PIPE, timeout=30)
        names = subprocess.check_output(
            ["git", "-C", str(upstream), "diff", "--no-ext-diff", "--no-textconv", "--name-only", "-z", "HEAD", "--"],
            stderr=subprocess.PIPE, timeout=30)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Cannot read pinned upstream commit {commit}: {exc.stderr.decode(errors='replace')}") from exc
    with tempfile.TemporaryDirectory(prefix="cipt_pair_pinned_") as directory:
        snapshot = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            for member in archive:
                rel = PurePosixPath(member.name)
                if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != "cipt":
                    raise ValueError("Invalid path in pinned source archive")
                destination = snapshot.joinpath(*rel.parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.extractfile(member).read())
                else:
                    raise ValueError("Expected only regular upstream source files")
        for name in ("__init__.py", "model.py", "templates.py", "engine.py", "losses.py"):
            if not (snapshot / "cipt" / name).is_file():
                raise ValueError(f"Pinned upstream package is incomplete: {name}")
        source = {"source_mode": "isolated_pinned_git_archive", "evaluated_commit": commit,
                  "snapshot_dir": str(snapshot), "working_tree_head": git_output(upstream, "rev-parse", "HEAD"),
                  "working_tree_changed_paths": [n.decode(errors="replace") for n in names.split(b"\0") if n],
                  "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
                  "snapshot_file_sha256": {p.relative_to(snapshot).as_posix(): sha256(p)
                                           for p in sorted(snapshot.rglob("*.py"))},
                  "note": "Evaluation uses only pinned Git objects; existing working-tree modifications are retained."}
        print(f"Upstream preflight: isolated official source {commit}; local changes retained: {len(source['working_tree_changed_paths'])} file(s)", flush=True)
        yield source


def import_upstream(source):
    snapshot = Path(source["snapshot_dir"])
    existing = sys.modules.get("cipt")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(snapshot):
        raise ValueError("A different cipt package is already imported; use a fresh Python process")
    sys.path.insert(0, str(snapshot))
    from cipt import build_cipt
    if not Path(sys.modules["cipt"].__file__).resolve().is_relative_to(snapshot):
        raise ValueError("The isolated pinned cipt package was not imported")
    return build_cipt


def validate_resume(previous, current):
    """Verify all fingerprints recorded by the original script before reuse."""
    fields = ("legacy_sha256", "upstream_sha256", "clip_sha256", "manifest_sha256",
              "classes", "coverage", "diagnostic_alphas", "torch_version")
    for field in fields:
        if field not in previous or previous[field] != current[field]:
            raise ValueError(f"Cannot reuse prior results: {field} changed")
    recorded = previous.get("source_sha256", {})
    if not recorded:
        raise ValueError("Prior results have no DomainBed source fingerprints")
    for filename, checksum in recorded.items():
        if current.get("source_sha256", {}).get(filename) != checksum:
            raise ValueError(f"Cannot reuse prior results: source changed: {filename}")


def cached_result(folder, kind, manifest, classes, alphas):
    import numpy as np
    folder = Path(folder)
    names = ("report.json", "summary.csv", "logits.npz", "features.npz")
    if not all((folder / name).is_file() for name in names):
        raise ValueError(f"Saved {kind} results are incomplete: {folder}")
    report = json.loads((folder / "report.json").read_text())
    if report.get("kind") != kind or len(report.get("input_tensor_sha256", "")) != 64:
        raise ValueError("Invalid cached model identity/input fingerprint")
    modes = ["native_TDA", "E_only", "LN_E"] + [f"alpha_{a:g}" for a in alphas]
    expected = {"labels": np.array([r["label"] for r in manifest]),
                "domains": np.array([r["domain"] for r in manifest]),
                "splits": np.array([r["split"] for r in manifest]),
                "relative_paths": np.array([r["relative_path"] for r in manifest]),
                "classes": np.array(classes)}
    with np.load(folder / "logits.npz", allow_pickle=False) as archive:
        if any(key not in archive or not np.array_equal(archive[key], value) for key, value in expected.items()):
            raise ValueError("Cached logits do not match the shared image manifest")
        values = {name: archive[name].copy() for name in modes}
    with np.load(folder / "features.npz", allow_pickle=False) as archive:
        if any(key not in archive or not np.array_equal(archive[key], value) for key, value in expected.items()):
            raise ValueError("Cached features do not match the shared image manifest")
        features = {name: archive[name].copy() for name in ("E_raw", "class_text", "text_increment")}
    if (features["E_raw"].ndim != 2 or features["E_raw"].shape[0] != len(manifest)
            or features["class_text"].shape != (len(classes), features["E_raw"].shape[1])
            or features["text_increment"].ndim != 3
            or features["text_increment"].shape[0] != len(classes)
            or features["text_increment"].shape[-1] != features["E_raw"].shape[1]
            or not all(np.isfinite(v).all() for v in features.values())):
        raise ValueError("Invalid cached feature arrays")
    if report.get("native_manual_max_abs_logit_error", float("inf")) > 1e-3:
        raise ValueError("Prior native/alpha parity check did not pass")
    rows = []
    for split in PARTITIONS:
        mask = expected["splits"] == split
        for mode, scores in values.items():
            if scores.shape != (len(manifest), len(classes)) or not np.isfinite(scores).all():
                raise ValueError("Invalid cached logits")
            metric = summarize(expected["labels"][mask], scores[mask], classes)
            saved = report["metrics"][split][mode]["pooled"]
            if metric["n"] != saved["n"] or metric["correct"] != saved["correct"]:
                raise ValueError("Cached report and logits disagree")
            per_domain = report["metrics"][split][mode]["per_domain"]
            macro = report["metrics"][split][mode]["domain_macro_accuracy"]
            for domain, m in [("POOLED", saved), *per_domain.items()]:
                rows.append({"model": kind, "split": split, "domain": domain, "mode": mode,
                             "n": m["n"], "accuracy": m["accuracy"],
                             "domain_macro_accuracy": macro if domain == "POOLED" else m["accuracy"],
                             "CE": m["CE_of_mean_logits"], "largest_predicted_class": m["largest_predicted_class"],
                             "largest_predicted_fraction": m["largest_predicted_fraction"]})
    print(f"RESUME: reusing verified {kind} results for {len(manifest)} images", flush=True)
    return report, values, rows, features


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write empty metrics table")
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def canonical(names):
    return [str(name).replace("_", " ") for name in names]


def check_legacy(hp):
    expected = {"cipt_objective": "legacy", "cipt_template_mode": "b5b",
                "cipt_use_tda": True, "cipt_use_de": True, "cipt_use_ind": True}
    for name, value in expected.items():
        if hp.get(name) != value:
            raise ValueError(f"Legacy checkpoint must be original B5B CIPT: {name}={hp.get(name)!r}")
    contrastive = bool(hp.get("cipt_use_contrastive", True)) and not bool(hp.get("cipt_pure", False))
    weight = float(hp.get("cipt_causal_contrastive_weight", hp.get("cipt_contrastive_weight", 0)))
    if contrastive and weight > 0:
        raise ValueError("Legacy checkpoint enables additional contrastive learning")
    if hp.get("cipt_selector_mode", "random") != "random":
        raise ValueError("Expected the original random-K template selector")
    if (float(hp.get("label_ratio", 1)) != 1 or hp.get("sample_d", False)
            or hp.get("mix", 0) or hp.get("indomain_test", False)):
        raise ValueError("Expected full-label original DomainBed in/out partitions")


def check_upstream(meta, legacy_args, target, classes):
    if meta.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Unexpected upstream CIPT commit")
    if meta.get("model_variant", "official") != "official":
        raise ValueError("consistent checkpoints are not original CIPT")
    if meta.get("no_swad") is not True or meta.get("source_val_for_model_selection") is not False:
        raise ValueError("Expected original final-epoch checkpoint without SWAD/IID selection")
    if meta.get("no_few_shot_sampling") is not True:
        raise ValueError("Expected full-data training, without few-shot sampling")
    for name in ("dataset", "trial_seed", "holdout_fraction", "seed"):
        if meta.get(name) != legacy_args.get(name):
            raise ValueError(f"Split/run mismatch in {name}: {legacy_args.get(name)!r} vs {meta.get(name)!r}")
    if int(meta.get("test_env", -1)) != target or canonical(meta.get("classes", [])) != canonical(classes):
        raise ValueError("Target domain or class order mismatch")
    if not math.isclose(float(meta.get("lr", 0)), 5e-5, rel_tol=0, abs_tol=1e-12):
        raise ValueError("This comparison expects the newly trained original lr=5e-5 checkpoint")


def summarize(labels, logits, classes):
    import numpy as np
    y, x = np.asarray(labels, dtype=np.int64), np.asarray(logits, dtype=np.float64)
    if not len(y) or x.shape != (len(y), len(classes)) or not np.isfinite(x).all():
        raise ValueError("Empty/nonfinite/misaligned classification results")
    if y.min() < 0 or y.max() >= len(classes):
        raise ValueError("Labels are outside the shared class order")
    pred = x.argmax(1)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    true_counts, counts = confusion.sum(1), confusion.sum(0)
    shifted = x - x.max(1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    wrong = x.copy(); wrong[np.arange(len(y)), y] = -np.inf
    margin = x[np.arange(len(y)), y] - wrong.max(1)
    recalls = np.divide(confusion.diagonal(), true_counts, out=np.zeros(len(classes)), where=true_counts > 0)
    return {
        "n": int(len(y)), "correct": int((pred == y).sum()), "accuracy": float((pred == y).mean()),
        "balanced_accuracy_present_classes": float(recalls[true_counts > 0].mean()),
        "CE_of_mean_logits": float(-logp[np.arange(len(y)), y].mean()),
        "mean_prediction_entropy": float(-(np.exp(logp) * logp).sum(1).mean()),
        "true_class_counts": dict(zip(classes, map(int, true_counts))),
        "predicted_class_counts": dict(zip(classes, map(int, counts))),
        "largest_predicted_class": classes[int(counts.argmax())],
        "largest_predicted_fraction": float(counts.max() / len(y)),
        "per_class_recall": {c: float(recalls[i]) if true_counts[i] else None for i, c in enumerate(classes)},
        "true_class_margin": {"mean": float(margin.mean()), "median": float(np.median(margin)),
                              "p10": float(np.quantile(margin, .1))},
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
    }


def transitions(labels, before, after):
    import numpy as np
    y = np.asarray(labels); a = np.asarray(before).argmax(1) == y; b = np.asarray(after).argmax(1) == y
    return {"both_correct": int((a & b).sum()), "corrected": int((~a & b).sum()),
            "regressed": int((a & ~b).sum()), "both_wrong": int((~a & ~b).sum()),
            "accuracy_gain_pp": float(100 * (b.mean() - a.mean()))}


def paired_cosines(before, after):
    import numpy as np
    a, b = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or not len(a) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Invalid paired feature arrays")
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    if np.any(an == 0) or np.any(bn == 0):
        raise ValueError("Zero-norm paired features")
    cosine = np.sum(a * b, axis=1) / (an * bn)
    return {"n": len(a), "mean_cosine": float(cosine.mean()), "median_cosine": float(np.median(cosine)),
            "p10_cosine": float(np.quantile(cosine, .1)), "before_norm_mean": float(an.mean()),
            "after_norm_mean": float(bn.mean())}


class ManifestImages:
    def __init__(self, manifest, splits):
        self.manifest, self.splits = manifest, splits

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest[index]
        sample = self.splits[row["partition_id"]][row["local_index"]]
        if int(sample["y"]) != row["label"]:
            raise ValueError("Dataset label changed after constructing the manifest")
        return {"x": sample["x"], "y": row["label"], "row_id": index}


def make_manifest(dataset, in_splits, out_splits, target, data_root, clip_basic, limit):
    sources = [i for i in range(len(dataset)) if i != target]
    parts = [(i, "source_train", in_splits[i][0]) for i in sources]
    parts += [(i, "source_val", out_splits[i][0]) for i in sources]
    parts += [(target, "target_in", in_splits[target][0]), (target, "target_out", out_splits[target][0])]
    records, split_objects, sizes = [], [], {p: {} for p in PARTITIONS}
    for part_id, (env, name, split) in enumerate(parts):
        split.transforms = {"x": clip_basic}
        split_objects.append(split)
        domain = dataset.environments[env]
        sizes[name][domain] = len(split)
        n = min(len(split), limit) if limit else len(split)
        if n == 0:
            raise ValueError(f"Empty split: {name}/{domain}")
        for i in range(n):
            underlying_index = int(split.keys[i])
            filename, label = split.underlying_dataset.samples[underlying_index]
            # Preserve lexical paths when a dataset subfolder is a symlink.
            filename = Path(filename).absolute()
            try:
                filename = filename.relative_to(data_root)
            except ValueError:
                raise ValueError("Image path is outside --data-dir")
            records.append({"row_id": len(records), "relative_path": filename.as_posix(),
                            "domain": domain, "split": name, "label": int(label),
                            "partition_id": part_id, "local_index": i, "underlying_index": underlying_index})
    if len({r["relative_path"] for r in records}) != len(records):
        raise ValueError("Overlapping or duplicate images in evaluation partitions")
    return records, ManifestImages(records, split_objects), sizes


def seed_all(seed, torch, np):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_torch(path, torch, trusted_legacy=False):
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = not trusted_legacy
    return torch.load(path, **options)


def load_upstream(source, checkpoint, clip_path, classes, device, torch):
    build_cipt = import_upstream(source)
    payload = load_torch(checkpoint, torch)
    meta, state = payload["metadata"], payload["trainable_state_dict"]
    model, _ = build_cipt(classes, backbone=str(clip_path), device=device, n_ctx=int(meta["n_ctx"]),
                         ctx_init="a photo of a", num_diverse_templates=int(meta["k"]),
                         num_heads=int(meta["num_heads"]), sample_templates=True)
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError(f"Trainable state mismatch: missing={expected - set(state)}, extra={set(state) - expected}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or expected.intersection(missing):
        raise ValueError("Upstream checkpoint was not fully restored")
    return model.to(device).eval(), meta


def evaluate(model, kind, imageset, manifest, classes, args, output):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    if kind == "domainbed":
        e_module, attention, norm = model.causal_decomposition, model.tda.attention, model.tda.layer_norm
    else:
        e_module = model.causal_adapter
        attention, norm = model.diversity_augmentation.attn, model.diversity_augmentation.norm
    captured = []
    def capture(_module, _inputs, value):
        captured.append((value[0] if kind == "domainbed" else value).detach())
    handle = e_module.register_forward_hook(capture)
    modes = ["native_TDA", "E_only", "LN_E"] + [f"alpha_{a:g}" for a in args.alphas]
    values = {m: [] for m in modes}
    input_hash = hashlib.sha256()
    error_max, disagree, cursor, norms, e_parts = 0.0, 0, 0, [], []
    try:
        with torch.inference_mode():
            if kind == "domainbed":
                anchors = model.text_features.class_features()
                bank = model.text_features.intervention_features(labels=None)
            else:
                anchors = model.encode_prompt_features()
                bank = model._select_diverse_features(labels=None)
            anchors = F.normalize(anchors.float(), dim=-1)
            bank = bank.detach().float()
            if bank.ndim != 3 or bank.shape[0] != len(classes) or bank.shape[-1] != anchors.shape[-1]:
                raise ValueError("Expected original class-conditioned [C,K,D] bank")
            c, k, dim = bank.shape
            p = bank.reshape(c * k, 1, dim)
            increment, _ = attention(torch.zeros_like(p), p, p, need_weights=False)
            increment = increment.reshape(c, k, dim)
            scale = model.clip_model.logit_scale.exp().detach().float()
            zero_z = F.normalize(norm(increment).float(), dim=-1)
            zero_scores = scale * torch.einsum("ckd,cd->ck", zero_z, anchors).mean(-1)
            loader = DataLoader(imageset, batch_size=args.batch_size, num_workers=args.num_workers,
                                shuffle=False, drop_last=False, pin_memory=str(args.device).startswith("cuda"))
            for batch in loader:
                row_ids = batch["row_id"].numpy()
                if not np.array_equal(row_ids, np.arange(cursor, cursor + len(row_ids))):
                    raise ValueError("Loader changed the shared image ordering")
                input_hash.update(batch["x"].contiguous().numpy().tobytes())
                x = batch["x"].to(args.device)
                captured.clear()
                native = model.predict(x) if kind == "domainbed" else model.predict(x, use_tda=True)
                if len(captured) != 1:
                    raise RuntimeError("Native prediction must compute E exactly once")
                e = captured[0]
                e_parts.append(e.float().cpu().numpy())
                scores = {"native_TDA": native,
                          "E_only": scale * F.normalize(e.float(), dim=-1) @ anchors.T,
                          "LN_E": scale * F.normalize(norm(e).float(), dim=-1) @ anchors.T}
                for alpha in args.alphas:
                    z = norm(e[:, None, None, :] + alpha * increment[None, :, :, :])
                    scores[f"alpha_{alpha:g}"] = scale * torch.einsum(
                        "bckd,cd->bck", F.normalize(z.float(), dim=-1), anchors).mean(-1)
                error = float((scores["alpha_1"] - native).abs().max())
                error_max = max(error_max, error)
                disagree += int((scores["alpha_1"].argmax(1) != native.argmax(1)).sum())
                if error > args.parity_atol:
                    raise RuntimeError(f"alpha=1/native logits mismatch {error}; diagnostic result is invalid")
                if not torch.allclose(scores["alpha_0"], scores["LN_E"], atol=args.parity_atol, rtol=0):
                    raise RuntimeError("alpha=0 must match LN(E)")
                for name, score in scores.items():
                    if not torch.isfinite(score).all():
                        raise ValueError("Nonfinite logits")
                    values[name].append(score.float().cpu().numpy())
                norms.extend((increment.norm(dim=-1).mean() / e.norm(dim=-1).clamp_min(1e-12)).cpu().tolist())
                cursor += len(row_ids)
                if cursor == len(manifest) or cursor % (args.batch_size * 20) == 0:
                    print(f"{kind}: evaluated {cursor}/{len(manifest)} shared images", flush=True)
            if cursor != len(manifest):
                raise ValueError("Incomplete image coverage")
            values = {name: np.concatenate(v) for name, v in values.items()}
            report = {"kind": kind, "input_tensor_sha256": input_hash.hexdigest(),
                      "native_manual_max_abs_logit_error": error_max,
                      "native_manual_prediction_disagreements": disagree,
                      "template_indices": list(range(k)),
                      "dtypes": {"clip_visual": str(model.clip_model.dtype), "bank": str(bank.dtype),
                                 "attention": str(attention.in_proj_weight.dtype)},
                      "zero_E_diagnostic": {"logits": zero_scores.cpu().tolist(),
                                            "predicted_class": classes[int(zero_scores.argmax())]},
                      "increment_norm_mean": float(increment.norm(dim=-1).mean()),
                      "trainable_parameters": {name: {"shape": list(p.shape), "dtype": str(p.dtype),
                           "norm": float(p.float().norm())} for name, p in model.named_parameters() if p.requires_grad},
                      "metrics": {}, "increment_over_E_norm": {}}
    finally:
        handle.remove()
    labels = np.array([r["label"] for r in manifest]); splits = np.array([r["split"] for r in manifest])
    domains = np.array([r["domain"] for r in manifest]); norms = np.array(norms)
    rows = []
    for split in PARTITIONS:
        mask = splits == split
        report["metrics"][split] = {}
        report["increment_over_E_norm"][split] = {"mean": float(norms[mask].mean()),
                                                  "median": float(np.median(norms[mask]))}
        for mode, logits in values.items():
            pooled = summarize(labels[mask], logits[mask], classes)
            per_domain = {d: summarize(labels[m], logits[m], classes)
                          for d in sorted(set(domains[mask])) if (m := (mask & (domains == d))).any()}
            macro = float(np.mean([v["accuracy"] for v in per_domain.values()]))
            report["metrics"][split][mode] = {"pooled": pooled, "domain_macro_accuracy": macro, "per_domain": per_domain}
            for domain, metric in [("POOLED", pooled), *per_domain.items()]:
                rows.append({"model": kind, "split": split, "domain": domain, "mode": mode,
                             "n": metric["n"], "accuracy": metric["accuracy"],
                             "domain_macro_accuracy": macro if domain == "POOLED" else metric["accuracy"],
                             "CE": metric["CE_of_mean_logits"],
                             "largest_predicted_class": metric["largest_predicted_class"],
                             "largest_predicted_fraction": metric["largest_predicted_fraction"]})
    output.mkdir(parents=True)
    write_json(output / "report.json", report)
    write_csv(output / "summary.csv", rows)
    write_csv(output / "predictions.csv", ({"relative_path": r["relative_path"], "domain": r["domain"],
        "split": r["split"], "true_class": classes[r["label"]],
        **{m: classes[int(v[r["row_id"]].argmax())] for m, v in values.items()}} for r in manifest))
    np.savez_compressed(output / "logits.npz", labels=labels, domains=domains, splits=splits,
                        classes=np.array(classes), relative_paths=np.array([r["relative_path"] for r in manifest]), **values)
    features = {"E_raw": np.concatenate(e_parts), "class_text": anchors.cpu().numpy(),
                "text_increment": increment.cpu().numpy()}
    np.savez_compressed(output / "features.npz", labels=labels, domains=domains, splits=splits,
                        classes=np.array(classes), relative_paths=np.array([r["relative_path"] for r in manifest]), **features)
    return report, values, rows, features


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--legacy-checkpoint", required=True, type=Path)
    p.add_argument("--upstream-checkpoint", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--clip-path", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--resume", action="store_true", help="Verify and reuse completed results in the same output directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-per-split", type=int, default=0, help="0 = full; positive = incomplete smoke test")
    p.add_argument("--alphas", type=float, nargs="+", default=[0, .1, .25, .5, 1])
    p.add_argument("--parity-atol", type=float, default=1e-3)
    args = p.parse_args(argv)
    if args.batch_size < 1 or min(args.num_workers, args.max_per_split) < 0:
        p.error("Invalid batch size, workers, or sample limit")
    if not math.isfinite(args.parity_atol) or args.parity_atol <= 0:
        p.error("Invalid parity tolerance")
    if any(not math.isfinite(a) or a < 0 for a in args.alphas):
        p.error("Diagnostic alpha must be finite and nonnegative")
    args.alphas = sorted(set(args.alphas + [0.0, 1.0]))
    return args


def main(argv=None):
    args = parse_args(argv)
    for name in ("repo_root", "legacy_checkpoint", "upstream_checkpoint", "data_dir", "clip_path", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    root, out = args.repo_root, args.output_dir
    for path in (args.legacy_checkpoint, args.upstream_checkpoint, args.clip_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Choose a new output directory: {out}")
    if args.resume and not (out / "protocol.json").is_file():
        raise FileNotFoundError(f"No saved evaluation protocol to resume: {out}")
    if not (root / "domainbed" / "algorithms" / "cipt_env_kl.py").is_file():
        raise FileNotFoundError("--repo-root must be the DCCL/DCCL directory containing the legacy algorithms")
    with clean_upstream_snapshot(root) as source:
        run_review(args, source)


def run_review(args, source):
    root, out = args.repo_root, args.output_dir
    previous = json.loads((out / "protocol.json").read_text()) if args.resume else None
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from domainbed.algorithms import get_algorithm_class
    from domainbed.datasets import get_dataset
    from domainbed.datasets.transforms import clip_basic
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Run on the training server with CUDA available")
    import_upstream(source)  # Fail early, before any image evaluation.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    old = load_torch(args.legacy_checkpoint, torch, trusted_legacy=True)
    if not {"model_hparams", "model_dict", "args", "test_envs"}.issubset(old):
        raise ValueError("Not a native DomainBed checkpoint")
    hp = dict(old["model_hparams"])
    original_hp = dict(hp)
    saved = dict(old["args"]) if isinstance(old["args"], dict) else dict(vars(old["args"]))
    check_legacy(hp)
    targets = list(map(int, old["test_envs"]))
    if len(targets) != 1 or saved.get("algorithm", "CIPTDCCL") != "CIPTDCCL":
        raise ValueError("Expected one held-out domain and the original CIPTDCCL algorithm")
    if saved.get("source_envs") is not None or saved.get("target_env") is not None:
        raise ValueError("Filtered domain mappings are unsupported")
    if not {"dataset", "seed", "trial_seed", "holdout_fraction"}.issubset(saved):
        raise ValueError("Missing saved split metadata")
    classes = list(hp.get("cipt_class_names", []))
    new = load_torch(args.upstream_checkpoint, torch)
    if not {"metadata", "trainable_state_dict"}.issubset(new):
        raise ValueError("Not an upstream trainable checkpoint")
    meta = dict(new["metadata"]); del new
    check_upstream(meta, saved, targets[0], classes)
    hp["cipt_clip_path"] = str(args.clip_path)
    seed_all(int(saved["seed"]), torch, np)
    algorithm = get_algorithm_class("CIPTDCCL")
    split_args = argparse.Namespace(dataset=saved["dataset"], data_dir=args.data_dir,
        trial_seed=int(saved["trial_seed"]), holdout_fraction=float(saved["holdout_fraction"]),
        source_envs=None, target_env=None)
    dataset, ins, outs = get_dataset(targets, split_args, hp, algorithm)
    if not classes or len(classes) != dataset.num_classes or any(canonical(env.classes) != canonical(classes) for env in dataset):
        raise ValueError("Dataset/checkpoint class order mismatch")
    manifest, imageset, sizes = make_manifest(dataset, ins, outs, targets[0], args.data_dir, clip_basic, args.max_per_split)
    expected_sizes = meta["sizes"]
    for name in ("source_train", "source_val"):
        if sizes[name] != expected_sizes[name]:
            raise ValueError(f"Different image counts in {name}: {sizes[name]} vs {expected_sizes[name]}")
    for name in ("target_in", "target_out"):
        if sum(sizes[name].values()) != expected_sizes[name]:
            raise ValueError(f"Different image counts in {name}")
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    protocol = {"legacy_checkpoint": str(args.legacy_checkpoint), "legacy_sha256": sha256(args.legacy_checkpoint),
        "upstream_checkpoint": str(args.upstream_checkpoint), "upstream_sha256": sha256(args.upstream_checkpoint),
        "legacy_hparams": original_hp, "legacy_args": saved, "upstream_metadata": meta,
        "clip_path": str(args.clip_path), "clip_sha256": sha256(args.clip_path),
        "repository_commit": git_output(root, "rev-parse", "HEAD"), "torch_version": str(torch.__version__),
        "repository_dirty": bool(git_output(root, "status", "--porcelain", "--untracked-files=no")),
        "source_sha256": {str(p): sha256(root / p) for p in [
            Path("domainbed/algorithms/cipt_env_kl.py"), Path("domainbed/algorithms/cipt_dccl_ablation.py"),
            Path("domainbed/algorithms/cipt_modules.py"), Path("domainbed/algorithms/cipt_prompt.py"),
            Path("domainbed/datasets/__init__.py"), Path("domainbed/datasets/transforms.py")]},
        "script_sha256": sha256(__file__), "audit_revision": AUDIT_REVISION, "manifest_sha256": manifest_hash,
        "upstream_source": source,
        "coverage": "FULL_DATASET_SPLITS" if not args.max_per_split else "INCOMPLETE_SMOKE_TEST",
        "classes": classes, "sizes": sizes, "diagnostic_alphas": args.alphas,
        "notes": ["Each model retains its own native frozen CLIP dtype and learned prompts.",
                  "All partitions use deterministic CLIP evaluation preprocessing.",
                  "Neither target partition was used to train either model.",
                  "No checkpoint/alpha selection is performed on source or target data.",
                  "Alpha=0 is LN(E); alpha=1 is verified against native inference."]}
    if previous is not None:
        validate_resume(previous, protocol)
        backup = out / "protocol.before_resume.json"
        if not backup.exists():
            backup.write_bytes((out / "protocol.json").read_bytes())
        protocol["resumed_from_script_sha256"] = previous.get("script_sha256")
    write_csv(out / "manifest.csv", manifest)
    write_json(out / "protocol.json", protocol)
    reused = []
    if args.resume and (out / "domainbed").exists():
        old_report, old_scores, old_rows, old_features = cached_result(out / "domainbed", "domainbed", manifest, classes, args.alphas)
        reused.append("domainbed"); del old
    else:
        model = algorithm(dataset.input_shape, dataset.num_classes, len(dataset) - 1, hp)
        model.load_state_dict(old["model_dict"], strict=True); del old
        model.to(args.device).eval()
        old_report, old_scores, old_rows, old_features = evaluate(model, "domainbed", imageset, manifest, classes, args, out / "domainbed")
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    seed_all(int(meta["seed"]), torch, np)
    if args.resume and (out / "upstream_lr5e5").exists():
        new_report, new_scores, new_rows, new_features = cached_result(out / "upstream_lr5e5", "upstream_lr5e5", manifest, classes, args.alphas)
        reused.append("upstream_lr5e5")
    else:
        model, _ = load_upstream(source, args.upstream_checkpoint, args.clip_path, classes, args.device, torch)
        new_report, new_scores, new_rows, new_features = evaluate(model, "upstream_lr5e5", imageset, manifest, classes, args, out / "upstream_lr5e5")
    if old_report["input_tensor_sha256"] != new_report["input_tensor_sha256"]:
        raise RuntimeError("The two models received different preprocessed image tensors")
    labels = np.array([r["label"] for r in manifest]); splits = np.array([r["split"] for r in manifest])
    comparison = {}
    for split in PARTITIONS:
        mask = splits == split
        comparison[split] = {mode: transitions(labels[mask], old_scores[mode][mask], new_scores[mode][mask])
                             for mode in ("native_TDA", "E_only", "LN_E")}
    write_json(out / "paired_changes.json", comparison)
    feature_comparison = {split: paired_cosines(old_features["E_raw"][splits == split],
                                                new_features["E_raw"][splits == split]) for split in PARTITIONS}
    feature_comparison["class_text_anchors"] = paired_cosines(old_features["class_text"], new_features["class_text"])
    feature_comparison["note"] = "Native E can differ because of the frozen encoder dtype and/or learned adapter. E-only scores also depend on each model's learned class texts."
    write_json(out / "paired_features.json", feature_comparison)
    metrics_path = args.upstream_checkpoint.with_name("metrics.json")
    if metrics_path.is_file() and not args.max_per_split:
        recorded = json.loads(metrics_path.read_text())
        protocol["upstream_saved_metrics_recheck"] = {
            split: {"recorded": recorded[split],
                    "reevaluated": new_report["metrics"][split]["native_TDA"]["pooled"]["accuracy"],
                    "difference_pp": 100 * (new_report["metrics"][split]["native_TDA"]["pooled"]["accuracy"] - recorded[split])}
            for split in ("source_val", "target_in", "target_out") if split in recorded}
        for split, result in protocol["upstream_saved_metrics_recheck"].items():
            print(f"Saved-metric recheck {split}: delta={result['difference_pp']:+.6f} pp", flush=True)
        write_json(out / "protocol.json", protocol)
    write_csv(out / "comparison.csv", old_rows + new_rows)
    joined = []
    for i, row in enumerate(manifest):
        a = int(old_scores["native_TDA"][i].argmax()); b = int(new_scores["native_TDA"][i].argmax())
        joined.append({"relative_path": row["relative_path"], "domain": row["domain"], "split": row["split"],
            "true_class": classes[row["label"]], "domainbed_prediction": classes[a], "upstream_prediction": classes[b],
            "domainbed_correct": a == row["label"], "upstream_correct": b == row["label"]})
    write_csv(out / "paired_predictions.csv", joined)
    lines = ["# Saved CIPT checkpoint comparison", "", f"Coverage: {protocol['coverage']}; identical input tensor hashes verified.", "",
             "Accuracies below are pooled over images. Source domain-macro accuracies are also saved in comparison.csv.", "",
             "| Split | Model | E-only | LN(E) | Native TDA |", "|---|---|---:|---:|---:|"]
    for split in PARTITIONS:
        for label, report in [("DomainBed", old_report), ("Upstream lr5e-5", new_report)]:
            m = report["metrics"][split]
            vals = [f"{100*m[k]['pooled']['accuracy']:.2f}%" for k in ("E_only", "LN_E", "native_TDA")]
            lines.append(f"| {split} | {label} | " + " | ".join(vals) + " |")
    lines += ["", "Both target_in and target_out were unused for training. This audit does not select a target partition, model, or alpha.",
              "", "Inspect report.json for per-class counts, confusion matrices, margins, dtype, and alpha controls."]
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "status.json", {"status": "COMPLETE", "identical_input_tensors": True,
                                     "resumed_model_results": reused,
                                     "evaluated_images": len(manifest), "coverage": protocol["coverage"]})
    print("\n" + "\n".join(lines) + f"\n\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
