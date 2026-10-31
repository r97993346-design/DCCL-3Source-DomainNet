#!/usr/bin/env python3
"""Read-only full-split audit of native DomainBed legacy/B5B or B5A CIPT checkpoints.

Run with the same branch/environment used for native inference. All weights and
algorithm switches come from the checkpoint; no YAML defaults are substituted.
Targets are never used for model or hyperparameter selection. Alpha is a
post-training diagnostic multiplier, not an official CIPT training parameter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import subprocess
import sys


UPSTREAM_COMMIT = "a805d878acc7d79778d1ec1c1e4d73ba6aff334b"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(root, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
            text=True, timeout=15,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def check_original(hp):
    required = {
        "cipt_objective": "legacy",
        "cipt_use_tda": True, "cipt_use_de": True, "cipt_use_ind": True,
    }
    for key, expected in required.items():
        if hp.get(key) != expected:
            raise ValueError(f"Not the requested original CIPT: {key}={hp.get(key)!r}, expected {expected!r}")
    if hp.get("cipt_template_mode") not in ("b5b", "b5a"):
        raise ValueError("Expected class-conditioned b5b or subject80 b5a.")
    if hp.get("cipt_template_mode") == "b5a" and hp.get("cipt_neutral_subject", "subject") != "subject":
        raise ValueError("This comparison expects the subject80 B5A bank.")
    pure = bool(hp.get("cipt_pure", False))
    contrastive = bool(hp.get("cipt_use_contrastive", True)) and not pure
    weight = float(hp.get("cipt_causal_contrastive_weight", hp.get("cipt_contrastive_weight", 0.0)))
    if contrastive and weight > 0:
        raise ValueError("Checkpoint enables contrastive learning; this audit expects original CIPT.")
    if hp.get("cipt_use_aug_view", False) and not pure:
        raise ValueError("Checkpoint enables an augmented view; review its training protocol first.")
    if hp.get("cipt_selector_mode", "random") != "random":
        raise ValueError("Expected original random-K template selection.")
    if int(hp.get("cipt_k", 0)) not in range(1, 81):
        raise ValueError("Expected 1 <= cipt_k <= 80.")


def summarize_arrays(labels, logits, classes):
    """Pure NumPy metrics; rows are true classes, columns are predictions."""
    import numpy as np
    y, x = np.asarray(labels, dtype=np.int64), np.asarray(logits, dtype=np.float64)
    if x.shape != (len(y), len(classes)) or not len(y) or not np.isfinite(x).all():
        raise ValueError("Invalid/empty labels or logits.")
    if y.min() < 0 or y.max() >= len(classes):
        raise ValueError("Labels outside class order.")
    pred = x.argmax(1)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    count = confusion.sum(1)
    recall = np.divide(confusion.diagonal(), count, out=np.zeros(len(classes)), where=count > 0)
    shifted = x - x.max(1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    predicted_counts = confusion.sum(0)
    return {
        "n": int(len(y)), "correct": int((pred == y).sum()),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy_present_classes": float(recall[count > 0].mean()),
        "CE_of_mean_logits": float(-logp[np.arange(len(y)), y].mean()),
        "mean_prediction_entropy": float(-(np.exp(logp) * logp).sum(1).mean()),
        "true_class_counts": dict(zip(classes, map(int, count))),
        "predicted_class_counts": dict(zip(classes, map(int, predicted_counts))),
        "largest_predicted_class": classes[int(predicted_counts.argmax())],
        "largest_predicted_fraction": float(predicted_counts.max() / len(y)),
        "per_class_recall": {c: float(recall[i]) if count[i] else None for i, c in enumerate(classes)},
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
    }


def make_upstream_clone(model, classes, official_root, device):
    """Isolate head/scoring parity: identical FP32 CLIP, bank, and learned weights.

    This intentionally does not reproduce upstream's original GPU loader or
    training RNG. It cannot diagnose a precision/training difference by itself.
    """
    root = Path(official_root).expanduser().resolve() / "third_party" / "CIPT"
    if not (root / "cipt" / "model.py").is_file():
        raise FileNotFoundError(f"Missing pinned upstream source at {root}")
    commit = git_output(root, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Upstream submodule is {commit}, expected {UPSTREAM_COMMIT}")
    if git_output(root, "diff", "--name-only", "HEAD"):
        raise ValueError("Upstream tracked source has local modifications; use the pinned source for parity.")
    if "cipt.model" in sys.modules:
        loaded = Path(sys.modules["cipt.model"].__file__).resolve()
        if not loaded.is_relative_to(root):
            raise ValueError(f"Another cipt package is already imported: {loaded}")
    sys.path.insert(0, str(root))
    from cipt.model import CIPTModel
    from cipt.templates import IMAGENET_TEMPLATES
    from domainbed.algorithms.cipt_prompt import build_paired_generic_templates
    class_conditioned = model.cipt_template_mode == "b5b"
    templates = IMAGENET_TEMPLATES if class_conditioned else build_paired_generic_templates("subject")
    paired = CIPTModel(
        model.clip_model, classes, model.text_features._tokenize,
        n_ctx=model.text_features.prompt_learner.context.shape[0], templates=templates,
        num_diverse_templates=model.text_features.k, num_heads=8,
    ).to(device).eval()
    paired.causal_adapter.proj.load_state_dict(model.causal_decomposition.causal_adapter.state_dict())
    paired.spurious_adapter.proj.load_state_dict(model.causal_decomposition.spurious_adapter.state_dict())
    paired.diversity_augmentation.attn.load_state_dict(model.tda.attention.state_dict())
    paired.diversity_augmentation.norm.load_state_dict(model.tda.layer_norm.state_dict())
    import torch
    with torch.no_grad():
        paired.prompt_learner.ctx.copy_(model.text_features.prompt_learner.context)
        for name in ("token_prefix", "token_suffix", "tokenized_prompts"):
            getattr(paired.prompt_learner, name).copy_(getattr(model.text_features.prompt_learner, name))
    active_bank = model.text_features.b5b_text_bank if class_conditioned else model.text_features.b5a_text_bank
    paired.diverse_text_features = active_bank.detach().clone()
    return paired


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Directory containing domainbed/")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--clip-path", type=Path, help="Only relocates the frozen CLIP file; does not change training flags")
    p.add_argument("--output-dir", required=True, type=Path, help="New, empty result directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--alphas", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1])
    p.add_argument("--max-per-split", type=int, default=0, help="0=full split; positive values are SMOKE TESTS only")
    p.add_argument("--official-root", type=Path, help="Optional iid-swad DCCL/DCCL directory for same-weight upstream parity")
    p.add_argument("--parity-atol", type=float, default=1e-3)
    a = p.parse_args(argv)
    if a.batch_size < 1 or min(a.num_workers, a.max_per_split) < 0:
        p.error("Invalid batch-size, workers, or sample limit.")
    if any(not math.isfinite(v) or v < 0 for v in a.alphas) or a.parity_atol <= 0:
        p.error("Alpha must be finite and nonnegative; parity-atol must be positive.")
    a.alphas = sorted(set(a.alphas + [0.0, 1.0]))
    return a


def main(argv=None):
    args = parse_args(argv)
    root = args.repo_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if not (root / "domainbed" / "algorithms" / "cipt_env_kl.py").is_file():
        raise FileNotFoundError("--repo-root must contain this branch's domainbed/algorithms/cipt_env_kl.py")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output already contains files: {out}; choose a new directory.")
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from domainbed.algorithms import get_algorithm_class
    from domainbed.datasets import get_dataset
    from domainbed.datasets.transforms import clip_basic

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the training server or --device cpu.")
    kw = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kw["weights_only"] = False  # User's trusted local training checkpoint contains args/hparams.
    checkpoint = torch.load(checkpoint_path, **kw)
    if not {"model_hparams", "model_dict", "args", "test_envs"}.issubset(checkpoint):
        raise ValueError("Expected native DomainBed checkpoint with args/model_hparams/model_dict/test_envs.")
    hp, saved_args = dict(checkpoint["model_hparams"]), dict(checkpoint["args"])
    original_hp = dict(hp)
    check_original(hp)
    class_conditioned = hp["cipt_template_mode"] == "b5b"
    if saved_args.get("algorithm", "CIPTDCCL") != "CIPTDCCL":
        raise ValueError("Checkpoint algorithm is not CIPTDCCL.")
    targets = list(map(int, checkpoint["test_envs"]))
    if len(targets) != 1 or hp.get("indomain_test", False):
        raise ValueError("This audit supports one held-out domain and the original 80/20 split.")
    if saved_args.get("source_envs") is not None or saved_args.get("target_env") is not None:
        raise ValueError("Filtered source/target environments need a separate index mapping.")
    if not {"dataset", "trial_seed", "holdout_fraction"}.issubset(saved_args):
        raise ValueError("Checkpoint lacks the exact dataset split metadata.")
    if float(hp.get("label_ratio", 1)) != 1 or hp.get("sample_d", False) or hp.get("mix", 0):
        raise ValueError("Expected full-label, original single-view CIPT data.")
    if args.clip_path is not None:
        hp["cipt_clip_path"] = str(args.clip_path.expanduser().resolve())
    clip_path = Path(hp.get("cipt_clip_path", "")).expanduser()
    if not clip_path.is_file():
        raise FileNotFoundError("Use --clip-path to relocate the original local CLIP file.")
    seed = int(saved_args.get("seed", 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    algorithm_class = get_algorithm_class("CIPTDCCL")
    split_args = argparse.Namespace(
        dataset=saved_args["dataset"], data_dir=args.data_dir.expanduser().resolve(),
        trial_seed=int(saved_args["trial_seed"]), holdout_fraction=float(saved_args["holdout_fraction"]),
        source_envs=None, target_env=None,
    )
    dataset, in_splits, out_splits = get_dataset(targets, split_args, hp, algorithm_class)
    classes = list(hp.get("cipt_class_names", []))
    canonical = lambda names: [str(n).replace("_", " ") for n in names]
    if not classes or len(classes) != dataset.num_classes:
        raise ValueError("Checkpoint needs an explicit real class order.")
    for env in dataset:
        if canonical(env.classes) != canonical(classes):
            raise ValueError("Dataset class order differs from checkpoint.")
    model = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - 1, hp)
    model.load_state_dict(checkpoint["model_dict"], strict=True)
    del checkpoint
    device = torch.device(args.device)
    model.to(device).eval()
    paired = make_upstream_clone(model, classes, args.official_root, device) if args.official_root else None
    out.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        anchors = F.normalize(model.text_features.class_features().float(), dim=-1)
        selected_bank = model.text_features.intervention_features(labels=None).detach().float()
        if class_conditioned:
            if selected_bank.ndim != 3:
                raise ValueError("B5B must return candidate-specific contexts [C,K,D].")
            bank = selected_bank
        else:
            if selected_bank.ndim != 2:
                raise ValueError("B5A must return shared contexts [K,D].")
            bank = selected_bank[None, :, :].expand(len(classes), -1, -1)
        c, k, d = bank.shape
        if (c, k, d) != (len(classes), int(hp["cipt_k"]), anchors.shape[1]):
            raise ValueError("Unexpected candidate bank shape.")
        # Exactly one key per attention call. With dropout=0 its softmax is 1;
        # precompute the image-independent text increment, then verify below
        # that the resulting alpha=1 logits match native predict(x).
        p = bank.reshape(c * k, 1, d)
        increment, _ = model.tda.attention(torch.zeros_like(p), p, p, need_weights=False)
        increment = increment.reshape(c, k, d)
        scale = model.clip_model.logit_scale.exp().detach().float()
        zero_z = F.normalize(model.tda.layer_norm(increment).float(), dim=-1)
        zero_logits = scale * torch.einsum("ckd,cd->ck", zero_z, anchors).mean(-1)

    captures = []
    def capture_e(_module, _inputs, output):
        captures.append(output[0].detach())
    handle = model.causal_decomposition.register_forward_hook(capture_e)
    mode_names = ["native_TDA", "E_only", "LN_E"] + [f"alpha_{a:g}" for a in args.alphas]
    labels_all, env_all, split_all, paths_all = [], [], [], []
    logits_all = {mode: [] for mode in mode_names}
    oracle_by_env, ratios, split_records = {}, {}, []
    max_manual_error = max_upstream_error = 0.0
    upstream_disagreements = 0

    # Complete source partitions precede all target partitions; none selects weights.
    sources = [i for i in range(len(dataset)) if i not in targets]
    partitions = [(i, "source_train", in_splits[i][0]) for i in sources]
    partitions += [(i, "source_val", out_splits[i][0]) for i in sources]
    partitions += [(targets[0], "target_in", in_splits[targets[0]][0]),
                   (targets[0], "target_out", out_splits[targets[0]][0])]
    try:
        with torch.inference_mode():
            for env_id, split_name, split in partitions:
                domain = dataset.environments[env_id]
                # Measure the training partition in deterministic evaluation mode.
                split.transforms = {"x": clip_basic}
                keys = list(split.keys)
                selected = list(range(len(split)))
                if args.max_per_split:
                    selected = selected[:args.max_per_split]
                if not selected:
                    raise ValueError(f"Empty partition {domain}/{split_name}")
                split_records.append({
                    "domain": domain, "split": split_name, "full_n": len(split), "evaluated_n": len(selected),
                    "ordered_indices_sha256": hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
                })
                loader = DataLoader(Subset(split, selected), batch_size=args.batch_size,
                                    num_workers=args.num_workers, shuffle=False, drop_last=False,
                                    pin_memory=device.type == "cuda")
                local_logits = {mode: [] for mode in mode_names}
                local_labels, local_ratios, oracle_logits = [], [], []
                oracle_template_ce_sum = 0.0
                cursor = 0
                for batch in loader:
                    images, y = batch["x"].to(device), batch["y"].to(device)
                    captures.clear()
                    native = model.predict(images)  # Intentionally NO ground-truth argument.
                    if len(captures) != 1:
                        raise RuntimeError("Expected one native E computation per predict(x).")
                    e = captures[0]
                    if not torch.isfinite(e).all() or not torch.isfinite(native).all():
                        raise ValueError("Non-finite native features/logits.")
                    scores = {"native_TDA": native, "E_only": scale * F.normalize(e.float(), dim=-1) @ anchors.T,
                              "LN_E": scale * F.normalize(model.tda.layer_norm(e).float(), dim=-1) @ anchors.T}
                    for alpha in args.alphas:
                        z = model.tda.layer_norm(e[:, None, None, :] + alpha * increment[None, :, :, :])
                        scores[f"alpha_{alpha:g}"] = scale * torch.einsum(
                            "bckd,cd->bck", F.normalize(z.float(), dim=-1), anchors).mean(-1)
                    error = float((scores["alpha_1"] - native).abs().max())
                    max_manual_error = max(max_manual_error, error)
                    if error > args.parity_atol:
                        raise RuntimeError(f"Manual/native logits differ by {error}; stop before interpreting alpha probes.")
                    if not torch.allclose(scores["alpha_0"], scores["LN_E"], atol=args.parity_atol, rtol=0):
                        raise RuntimeError("Alpha=0 must equal LN(E).")
                    if paired is not None:
                        upstream = paired.predict(images, use_tda=True)
                        error = float((upstream - native).abs().max())
                        max_upstream_error = max(max_upstream_error, error)
                        upstream_disagreements += int((upstream.argmax(1) != native.argmax(1)).sum())
                        if error > args.parity_atol:
                            raise RuntimeError(f"Same-weight upstream/native mismatch {error}; inspect code before attributing training.")
                    local_ratios.extend((increment.norm(dim=-1).mean() / e.norm(dim=-1).clamp_min(1e-12)).cpu().tolist())
                    # True-label context is a source-only mechanism diagnostic.
                    if class_conditioned and split_name.startswith("source_"):
                        gold_z = model.tda.layer_norm(e[:, None, :] + increment[y])
                        gold_logits = scale * torch.einsum("bkd,cd->bkc", F.normalize(gold_z.float(), dim=-1), anchors)
                        oracle_template_ce_sum += float(F.cross_entropy(
                            gold_logits.reshape(-1, c), y[:, None].expand(-1, k).reshape(-1), reduction="sum"))
                        oracle_logits.append(gold_logits.mean(1).cpu().numpy())
                    for mode in mode_names:
                        local_logits[mode].append(scores[mode].float().cpu().numpy())
                    local_labels.extend(y.cpu().tolist())
                    for local_i in selected[cursor:cursor + len(y)]:
                        sample_path = Path(split.underlying_dataset.samples[keys[local_i]][0])
                        try:
                            sample_path = sample_path.relative_to(split_args.data_dir)
                        except ValueError:
                            pass
                        paths_all.append(str(sample_path))
                    cursor += len(y)
                n = len(local_labels)
                labels_all.extend(local_labels); env_all.extend([domain] * n); split_all.extend([split_name] * n)
                ratios[f"{split_name}/{domain}"] = {"all_candidate_mean_increment_over_E_norm_mean": float(np.mean(local_ratios)),
                                                     "p50": float(np.median(local_ratios)), "p95": float(np.quantile(local_ratios, .95))}
                for mode in mode_names:
                    logits_all[mode].append(np.concatenate(local_logits[mode]))
                if oracle_logits:
                    oracle_by_env[f"{split_name}/{domain}"] = {
                        "NOT_VALID_LABEL_FREE_ACCURACY": summarize_arrays(local_labels, np.concatenate(oracle_logits), classes),
                        "mean_per_template_CE_matching_training": oracle_template_ce_sum / (n * k),
                    }
                print(f"{split_name:12s} {domain:16s} n={n:5d} native={summarize_arrays(local_labels, np.concatenate(local_logits['native_TDA']), classes)['accuracy']:.4%}", flush=True)
    finally:
        handle.remove()

    y = np.asarray(labels_all, dtype=np.int64)
    domains, splits = np.asarray(env_all), np.asarray(split_all)
    logits_all = {mode: np.concatenate(parts) for mode, parts in logits_all.items()}
    report = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_hparams_original": original_hp,
        "resolved_clip_path": str(clip_path), "resolved_clip_sha256": sha256(clip_path),
        "checkpoint_args": saved_args, "test_envs": targets, "classes": classes,
        "repository_commit": git_output(root, "rev-parse", "HEAD"),
        "repository_dirty": bool(git_output(root, "status", "--porcelain", "--untracked-files=no")),
        "source_sha256": {p: sha256(root / p) for p in [
            "domainbed/algorithms/algorithms.py", "domainbed/algorithms/cipt_env_kl.py",
            "domainbed/algorithms/cipt_dccl_ablation.py", "domainbed/algorithms/cipt_dccl_component_ablation.py",
            "domainbed/algorithms/cipt_modules.py", "domainbed/algorithms/cipt_prompt.py",
            "domainbed/datasets/__init__.py", "domainbed/datasets/transforms.py"]},
        "torch_version": str(torch.__version__), "device": str(device),
        "dtypes": {"clip_visual": str(model.clip_model.dtype), "bank": str(bank.dtype),
                   "context": str(model.text_features.prompt_learner.context.dtype),
                   "attention": str(model.tda.attention.in_proj_weight.dtype)},
        "coverage": "FULL_DATASET_SPLITS" if not args.max_per_split else "SMOKE_TEST_PARTIAL_DO_NOT_REPORT_AS_FULL_ACCURACY",
        "partitions": split_records, "alpha_values_diagnostic_only": args.alphas,
        "template_indices": list(range(k)), "native_manual_max_abs_logit_error": max_manual_error,
        "active_template_mode": hp["cipt_template_mode"],
        "class_conditioned_context": class_conditioned,
        "native_selected_bank_shape": list(selected_bank.shape),
        "same_weight_upstream_parity": None if paired is None else {
            "upstream_commit": UPSTREAM_COMMIT, "shared_clip_and_saved_bank": True,
            "max_abs_logit_error": max_upstream_error, "prediction_disagreements": upstream_disagreements,
            "note": "Same weights/FP32 CLIP/bank; checks scoring implementation, not original training or GPU precision."},
        "zero_E_diagnostic": {"predicted_class": classes[int(zero_logits.argmax())], "logits": zero_logits.cpu().tolist()},
        "increment_norm_mean": float(increment.norm(dim=-1).mean()),
        "increment_norm_mean_by_class": {name: float(increment[i].norm(dim=-1).mean()) for i, name in enumerate(classes)},
        "trainable_parameter_statistics": {name: {
            "shape": list(p.shape), "dtype": str(p.dtype), "norm": float(p.detach().float().norm()),
            "rms": float(p.detach().float().square().mean().sqrt()),
            "all_finite": bool(torch.isfinite(p).all()),
        } for name, p in model.named_parameters() if p.requires_grad},
        "increment_over_E_norm": ratios,
        "source_gold_context_DIAGNOSTIC_NOT_VALID_ACCURACY": oracle_by_env,
        "metrics": {},
        "notes": ["All inference metrics use native predict(x), E, or explicitly named post-training controls; no labels choose target contexts.",
                  "source_train is the actual training partition in eval mode; source_val was held out from training.",
                  "target_in and target_out were both unused for training; target_in is not a target training set.",
                  "No model/alpha selection or parameter optimization is performed; report target_in/out separately.",
                  "Alpha=0 is LN(E), which can differ from E-only; alpha is not an official CIPT hyperparameter."],
    }
    summary_rows = []
    for split_name in ("source_train", "source_val", "target_in", "target_out"):
        split_mask = splits == split_name
        report["metrics"][split_name] = {}
        for mode, values in logits_all.items():
            per_domain = {domain: summarize_arrays(y[mask], values[mask], classes)
                          for domain in sorted(set(domains[split_mask]))
                          if (mask := (split_mask & (domains == domain))).any()}
            pooled = summarize_arrays(y[split_mask], values[split_mask], classes)
            macro = float(np.mean([v["accuracy"] for v in per_domain.values()]))
            report["metrics"][split_name][mode] = {"pooled": pooled, "domain_macro_accuracy": macro, "per_domain": per_domain}
            summary_rows.append({"split": split_name, "domain": "POOLED", "mode": mode,
                                 "n": pooled["n"], "accuracy": pooled["accuracy"],
                                 "domain_macro_accuracy": macro, "CE": pooled["CE_of_mean_logits"],
                                 "largest_predicted_class": pooled["largest_predicted_class"],
                                 "largest_predicted_fraction": pooled["largest_predicted_fraction"]})
            for domain, m in per_domain.items():
                summary_rows.append({"split": split_name, "domain": domain, "mode": mode,
                                     "n": m["n"], "accuracy": m["accuracy"], "domain_macro_accuracy": m["accuracy"],
                                     "CE": m["CE_of_mean_logits"], "largest_predicted_class": m["largest_predicted_class"],
                                     "largest_predicted_fraction": m["largest_predicted_fraction"]})
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0])); w.writeheader(); w.writerows(summary_rows)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["relative_path", "domain", "split", "true_class", *mode_names])
        pred = {m: x.argmax(1) for m, x in logits_all.items()}
        for i, label in enumerate(y):
            w.writerow([paths_all[i], domains[i], splits[i], classes[int(label)], *[classes[int(pred[m][i])] for m in mode_names]])
    np.savez_compressed(out / "logits.npz", labels=y, domains=domains, splits=splits,
                        classes=np.asarray(classes), **logits_all)
    print(f"Saved read-only audit: {out}", flush=True)


if __name__ == "__main__":
    main()
