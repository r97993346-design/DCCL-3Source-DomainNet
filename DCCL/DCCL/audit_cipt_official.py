#!/usr/bin/env python3
"""Audit existing original CIPT checkpoints without retraining.

Run from DCCL/DCCL with the same Python environment as the training script.
No checkpoint or repository file is modified. Results are printed as JSON.
Compare E-only, the original LayerNorm applied to E, and native TDA using
exactly the same images, learned class texts, weights, and template indices.
The optional optimizer probe changes in-memory weights once, then restores them.
Optional --alphas and --probe-zero-image are evaluation-only interventions;
alpha=1 is the original model and alpha=0 removes only the attention increment.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def history_summary(path):
    if not path.is_file():
        return {"available": False}
    rows = json.loads(path.read_text(encoding="utf-8"))
    valid = [r for r in rows if isinstance(r.get("source_val"), (int, float))]
    return {
        "available": True,
        "first": rows[0] if rows else None,
        "best_source_val": max(valid, key=lambda r: r["source_val"]) if valid else None,
        "last": rows[-1] if rows else None,
    }


def alpha_mode_name(alpha):
    return "tda_alpha_" + format(alpha, ".12g")


def predict_scaled_tda(model, images, template_indices, alpha):
    """Scale the attention output, retaining native candidate scoring and LN."""
    def scale_increment(_module, _inputs, output):
        return (output[0] * alpha, output[1])
    handle = model.diversity_augmentation.attn.register_forward_hook(scale_increment)
    try:
        return model.predict(images, use_tda=True, template_indices=template_indices)
    finally:
        handle.remove()


def predict_zero_image_tda(model, images, template_indices):
    """Set E to zero at its source, affecting BOTH attention Query and residual."""
    def zero_causal_features(_module, _inputs, output):
        return torch.zeros_like(output)
    handle = model.causal_adapter.register_forward_hook(zero_causal_features)
    try:
        return model.predict(images, use_tda=True, template_indices=template_indices)
    finally:
        handle.remove()


def diagnose_split(model, batches, fixed_text, num_classes, cipt_loss, beta, gamma,
                   template_indices, alphas=None, probe_zero_image=False):
    with torch.no_grad():
        return _diagnose_split(model, batches, fixed_text, num_classes, cipt_loss,
                               beta, gamma, template_indices, alphas, probe_zero_image)


def _diagnose_split(model, batches, fixed_text, num_classes, cipt_loss, beta, gamma,
                    template_indices, alphas=None, probe_zero_image=False):
    model.eval()
    alphas = [] if alphas is None else list(alphas)
    modes = ["clip_fixed_prompt", "clip_learned_prompt", "e_only", "e_layernorm_only",
             "tda_inference", "tda_true_label_ORACLE"]
    modes += [alpha_mode_name(alpha) for alpha in alphas]
    if probe_zero_image:
        modes.append("tda_zero_image")
    totals = {name: {"correct": 0, "ce_sum": 0.0,
                     "score_sum": torch.zeros(num_classes, dtype=torch.float64),
                     "confusion": torch.zeros(num_classes, num_classes, dtype=torch.long)}
              for name in modes}
    losses_sum = {}
    fusion_sum = {"rows": 0, "image_norm": 0.0, "text_increment_norm": 0.0,
                  "text_to_image_norm_ratio": 0.0}
    def capture_native_fusion(_module, inputs, output):
        # The hook is active ONLY during native predict(), never the oracle.
        # It reads tensors and returns None, leaving the output unchanged.
        image = inputs[0].detach().float()
        increment = output[0].detach().float()
        image_norm = image.norm(dim=-1).reshape(-1)
        increment_norm = increment.norm(dim=-1).reshape(-1)
        fusion_sum["rows"] += image_norm.numel()
        fusion_sum["image_norm"] += float(image_norm.sum())
        fusion_sum["text_increment_norm"] += float(increment_norm.sum())
        fusion_sum["text_to_image_norm_ratio"] += float(
            (increment_norm / image_norm.clamp_min(1e-12)).sum())
    n = 0
    device = next(model.parameters()).device
    for images, labels in batches:
        images, labels = images.to(device), labels.to(device)
        out = model(images, labels, template_indices=template_indices)
        handle = model.diversity_augmentation.attn.register_forward_hook(capture_native_fusion)
        try:
            native_tda = model.predict(images, use_tda=True, template_indices=template_indices)
        finally:
            handle.remove()
        norm_only = model.diversity_augmentation.norm(out.causal_features.float())
        scores = {
            "clip_fixed_prompt": model.logit_scale * out.image_features @ fixed_text.t(),
            "clip_learned_prompt": model.logit_scale * out.image_features @ out.text_features.t(),
            "e_only": out.causal_logits,
            "e_layernorm_only": model._logits(norm_only, out.text_features),
            "tda_inference": native_tda,
            "tda_true_label_ORACLE": out.interventional_logits.mean(dim=1),
        }
        for alpha in alphas:
            key = alpha_mode_name(alpha)
            # alpha=1 reuses the native result, so it cannot drift by re-encoding.
            scores[key] = native_tda if alpha == 1.0 else predict_scaled_tda(
                model, images, template_indices, alpha)
            if alpha == 0.0 and not torch.allclose(
                    scores[key], scores["e_layernorm_only"], rtol=1e-5, atol=1e-4):
                raise RuntimeError("alpha=0 does not reproduce LayerNorm(E); inspect the upstream model version")
        if probe_zero_image:
            scores["tda_zero_image"] = predict_zero_image_tda(model, images, template_indices)
        for mode, logits in scores.items():
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite logits in {mode}")
            pred = logits.argmax(dim=-1)
            totals[mode]["correct"] += int((pred == labels).sum())
            totals[mode]["ce_sum"] += float(F.cross_entropy(logits, labels, reduction="sum"))
            totals[mode]["score_sum"] += logits.detach().double().sum(dim=0).cpu()
            pairs = labels * num_classes + pred
            counts = torch.bincount(pairs, minlength=num_classes ** 2)
            totals[mode]["confusion"] += counts.cpu().reshape(num_classes, num_classes)
        losses = cipt_loss(out.interventional_logits, out.causal_logits, out.spurious_logits,
                           out.causal_features, out.spurious_features, labels,
                           beta=beta, gamma=gamma)
        for name in ["loss", "classification", "causal_ce", "spurious_kl", "independence"]:
            losses_sum[name] = losses_sum.get(name, 0.0) + float(getattr(losses, name)) * len(labels)
        n += len(labels)
    if n == 0:
        raise RuntimeError("Empty diagnostic split")
    result = {"num_samples": n, "modes": {}}
    for mode, stats in totals.items():
        confusion = stats["confusion"]
        result["modes"][mode] = {
            "accuracy_pct": round(100 * stats["correct"] / n, 6),
            "cross_entropy": stats["ce_sum"] / n,
            "true_class_counts": confusion.sum(1).tolist(),
            "predicted_class_counts": confusion.sum(0).tolist(),
            "confusion_true_rows_predicted_columns": confusion.tolist(),
            "mean_class_logits": (stats["score_sum"] / n).tolist(),
        }
    result["alpha_diagnostics"] = {
        "alphas": alphas, "zero_image_enabled": probe_zero_image,
        "scaling_location": "attention output before residual addition and original LayerNorm",
        "zero_image_location": "causal adapter output: zeros both image Query and image residual",
        "note": "Frozen-checkpoint diagnostics on source data; these are not retrained models or target-domain model selection.",
    }
    result["losses_with_fixed_templates"] = {k: v / n for k, v in losses_sum.items()}
    # This is a checkpoint diagnostic, not a claim that large ratios are a bug.
    rows = fusion_sum.pop("rows")
    result["native_fusion"] = {
        "num_image_candidate_template_rows": rows,
        **{key + "_mean": value / rows if rows else None
           for key, value in fusion_sum.items()},
    }
    result["template_indices"] = template_indices.detach().cpu().tolist()
    result["diagnostic_note"] = (
        "ORACLE selects TDA contexts using true labels and is NOT a deployable accuracy. "
        "A/B/C isolate evaluation paths; their gaps alone do not prove an upstream bug.")
    return result


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary_rows(report):
    rows = []
    for result in report["results"]:
        for split in ("source_train_sample", "source_val"):
            item = result[split]
            modes = item["modes"]
            rows.append({
                "target": result["target"], "split": split, "samples": item["num_samples"],
                "CLIP_fixed_pct": modes["clip_fixed_prompt"]["accuracy_pct"],
                "CLIP_learned_pct": modes["clip_learned_prompt"]["accuracy_pct"],
                "A_E_only_pct": modes["e_only"]["accuracy_pct"],
                "B_LayerNorm_E_pct": modes["e_layernorm_only"]["accuracy_pct"],
                "C_native_TDA_pct": modes["tda_inference"]["accuracy_pct"],
                "ORACLE_invalid_as_test_pct": modes["tda_true_label_ORACLE"]["accuracy_pct"],
                "B_minus_A_pp": round(modes["e_layernorm_only"]["accuracy_pct"] - modes["e_only"]["accuracy_pct"], 6),
                "C_minus_B_pp": round(modes["tda_inference"]["accuracy_pct"] - modes["e_layernorm_only"]["accuracy_pct"], 6),
            })
            for mode, stats in modes.items():
                if not (mode.startswith("tda_alpha_") or mode == "tda_zero_image"):
                    continue
                counts = stats["predicted_class_counts"]
                index = max(range(len(counts)), key=counts.__getitem__)
                rows[-1][mode + "_pct"] = stats["accuracy_pct"]
                rows[-1][mode + "_dominant_class"] = result["classes"][index]
                rows[-1][mode + "_dominant_prediction_pct"] = round(
                    100 * counts[index] / item["num_samples"], 6)
    return rows


def print_summary(rows):
    print("\nSame-checkpoint source-only audit (percent)", file=sys.stderr)
    print("target / split                  n       E-only      LN(E)        TDA       ORACLE*", file=sys.stderr)
    for row in rows:
        label = row["target"] + "/" + row["split"]
        print(f"{label:<31} {row['samples']:5d} {row['A_E_only_pct']:11.3f} "
              f"{row['B_LayerNorm_E_pct']:10.3f} {row['C_native_TDA_pct']:10.3f} "
              f"{row['ORACLE_invalid_as_test_pct']:12.3f}", file=sys.stderr)
    print("* ORACLE uses true labels to select text; diagnostic only, not valid test accuracy.\n", file=sys.stderr)
    for row in rows:
        probes = [key[:-4] for key in row if key.endswith("_pct")
                  and not key.endswith("_dominant_prediction_pct")
                  and (key.startswith("tda_alpha_") or key == "tda_zero_image_pct")]
        if not probes:
            continue
        print(f"{row['target']} / {row['split']}: attention-increment probes", file=sys.stderr)
        print("mode                         accuracy%   dominant prediction   share%", file=sys.stderr)
        for mode in probes:
            print(f"{mode:<29} {row[mode + '_pct']:9.3f}   "
                  f"{row[mode + '_dominant_class']:<21} "
                  f"{row[mode + '_dominant_prediction_pct']:7.3f}", file=sys.stderr)


def optimizer_probe(model, batch, make_optimizer, cipt_loss, cfg):
    device = next(model.parameters()).device
    images, labels = [x.to(device) for x in batch]
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    before = {name: p.detach().clone() for name, p in parameters.items()}
    was_training = model.training
    optimizer = make_optimizer(model, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    try:
        model.train()
        out = model(images, labels)
        losses = cipt_loss(out.interventional_logits, out.causal_logits, out.spurious_logits,
                           out.causal_features, out.spurious_features, labels,
                           beta=cfg["beta"], gamma=cfg["gamma"])
        optimizer.zero_grad(set_to_none=True)
        losses.loss.backward()
        gradients = {}
        for name, p in parameters.items():
            g = p.grad
            gradients[name] = {"has_gradient": g is not None,
                               "finite": bool(torch.isfinite(g).all()) if g is not None else None,
                               "norm": finite_float(g.float().norm()) if g is not None else None}
        if not all(s["finite"] is not False for s in gradients.values()):
            return {"loss": finite_float(losses.loss.detach()), "gradients": gradients,
                    "step_skipped": "non-finite gradient"}
        optimizer.step()
        updates = {name: finite_float((p.detach() - before[name]).float().norm())
                   for name, p in parameters.items()}
        return {"loss": finite_float(losses.loss.detach()), "gradients": gradients,
                "fresh_adam_step_update_norms": updates,
                "weights_finite_after_step": all(bool(torch.isfinite(p).all()) for p in parameters.values()),
                "note": "Fresh Adam state, not a resumed training step; weights restored in memory."}
    finally:
        with torch.no_grad():
            for name, p in parameters.items():
                p.copy_(before[name])
                p.grad = None
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--results-root", type=Path, default=Path("train_output/cipt_official_domainbed/PACS"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--clip-path", default="")
    parser.add_argument("--test-envs", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trial-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--template-indices", nargs="+", type=int,
                        help="Exactly K distinct template indices; default is native first K")
    parser.add_argument("--alphas", nargs="+", type=float,
                        help="Evaluation-only attention-increment scales, e.g. 0 0.1 0.25 0.5 1")
    parser.add_argument("--probe-zero-image", action="store_true",
                        help="Also run native TDA after zeroing the causal image features")
    parser.add_argument("--output", type=Path,
                        help="Write JSON plus a sibling .csv summary; refuses existing files")
    parser.add_argument("--check-gradients", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_train_samples < 1:
        parser.error("batch-size/max-train-samples must be positive; num-workers nonnegative")
    if len(set(args.test_envs)) != len(args.test_envs):
        parser.error("test-envs must be distinct")
    if args.template_indices is not None and (
            min(args.template_indices) < 0 or len(set(args.template_indices)) != len(args.template_indices)):
        parser.error("template-indices must be distinct nonnegative integers")
    if args.alphas is not None and (
            any(not math.isfinite(alpha) or alpha < 0 for alpha in args.alphas)
            or len(set(args.alphas)) != len(args.alphas)
            or len(set(map(alpha_mode_name, args.alphas))) != len(args.alphas)):
        parser.error("alphas must be distinct finite nonnegative numbers")
    repo = args.repo_dir.resolve()
    output = None
    if args.output is not None:
        output = args.output if args.output.is_absolute() else repo / args.output
        output = output.resolve()
        if output.suffix.lower() != ".json":
            parser.error("--output must end in .json")
        for destination in (output, output.with_suffix(".csv")):
            if destination.exists():
                parser.error(f"Output already exists; choose a new name: {destination}")
    entry = repo / "train_cipt_official_domainbed.py"
    if not entry.is_file():
        parser.error(f"Run from DCCL/DCCL or set --repo-dir; missing {entry}")
    # --help and report helpers work without loading the training environment.
    global np, torch, F, DataLoader, Subset
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Subset
    except ImportError as error:
        parser.error(f"Use the same Python environment as CIPT training: {error}")
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("cipt_audited_entry", entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    build, _, _, make_optimizer, _ = module.load_official_api()
    from cipt.losses import cipt_loss

    root = args.results_root if args.results_root.is_absolute() else repo / args.results_root
    checkpoints = sorted(root.glob(f"target_*/seed_{args.seed}_trial_{args.trial_seed}/final_trainable.pth"))
    chosen = []
    for path in checkpoints:
        cfg = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        if cfg["test_env"] in args.test_envs:
            chosen.append((path, cfg))
    missing = set(args.test_envs) - {cfg["test_env"] for _, cfg in chosen}
    if missing:
        parser.error(f"No checkpoints for test-envs {sorted(missing)} under {root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"audit_version": "2026-10-31-alpha-zero-image", "torch_version": torch.__version__,
              "device": str(device), "results": [], "model_variant": "upstream_CIPT",
              "note": "Source splits only. ORACLE uses true labels to choose TDA texts; it is NOT valid test accuracy."}
    for path, cfg in chosen:
        if "consistent" in str(path).lower() or cfg.get("model_variant") == "consistent":
            parser.error("Use original cipt_official_domainbed checkpoints for this audit, not consistent-model results")
        print(f"Auditing {path}", file=sys.stderr, flush=True)
        seed_all(cfg["seed"])
        split_args = argparse.Namespace(dataset=cfg["dataset"], data_dir=args.data_dir,
                                        trial_seed=cfg["trial_seed"], holdout_fraction=cfg["holdout_fraction"])
        with contextlib.redirect_stdout(sys.stderr):
            dataset, names, train, val, _, _, sizes = module.make_splits(split_args, cfg["test_env"])
            model, _ = build(names, backbone=args.clip_path or cfg["backbone_or_path"], device=device,
                             n_ctx=cfg["n_ctx"], ctx_init="a photo of a", num_heads=cfg["num_heads"],
                             num_diverse_templates=cfg["k"], sample_templates=True)
        if names != cfg["classes"]:
            raise RuntimeError("Class names/order differ from the saved configuration")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["metadata"]["classes"] != names or saved["metadata"]["test_env"] != cfg["test_env"]:
            raise RuntimeError("Checkpoint metadata does not match its config.json")
        state = saved["trainable_state_dict"]
        expected = {name for name, p in model.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise RuntimeError(f"Trainable checkpoint mismatch: missing={sorted(expected-set(state))}; extra={sorted(set(state)-expected)}")
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise RuntimeError("Checkpoint contains non-finite trainable weights")
        model.load_state_dict(state, strict=False)
        model.eval()
        if not model.class_conditioned_templates:
            raise RuntimeError("This audit expects the original class-conditioned template bank")
        available = model.diverse_text_features.shape[1]
        selected = list(range(min(cfg["k"], available))) if args.template_indices is None else args.template_indices
        if len(selected) != min(cfg["k"], available) or max(selected) >= available:
            raise RuntimeError(f"Need exactly {min(cfg['k'], available)} template indices in [0, {available - 1}]")
        template_indices = torch.tensor(selected, dtype=torch.long, device=device)
        with torch.no_grad():
            tokens = model.tokenize([f"a photo of a {name}." for name in names]).to(device)
            fixed_text = F.normalize(model.clip_model.encode_text(tokens).float(), dim=-1)
            learned = model.encode_prompt_features()
            mask = ~torch.eye(len(names), dtype=torch.bool, device=device)
            sim = learned @ learned.t()
            prompt_cos = float(sim[mask].mean()) if len(names) > 1 else None
        ids = torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg["seed"]))
        train_sample = Subset(train, ids[:args.max_train_samples].tolist())
        def batches(split):
            return DataLoader(split, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
        item = {"checkpoint": str(path), "checkpoint_sha256": checkpoint_sha256(path),
                "target": cfg["environment"], "classes": names,
                "class_names_are_numeric": all(name.isdigit() for name in names),
                "sizes": sizes, "history": history_summary(path.parent / "history.json"),
                "learned_text_mean_off_diagonal_cosine": prompt_cos,
                "clip_parameters_frozen": all(not p.requires_grad for p in model.clip_model.parameters()),
                "config": {k: cfg[k] for k in ["lr", "epochs", "beta", "gamma", "k", "n_ctx", "num_heads"]}}
        item["template_indices"] = selected
        item["template_texts"] = [model.templates[index] for index in selected]
        item["upstream_commit_saved_in_config"] = cfg.get("upstream_commit")
        for name, split in [("source_train_sample", train_sample), ("source_val", val)]:
            print(f"  Evaluating {name} ({len(split)} samples)", file=sys.stderr, flush=True)
            item[name] = diagnose_split(model, batches(split), fixed_text, len(names),
                                        cipt_loss, cfg["beta"], cfg["gamma"], template_indices,
                                        args.alphas, args.probe_zero_image)
        if args.check_gradients:
            item["optimizer_probe"] = optimizer_probe(model, next(iter(batches(train_sample))),
                                                       make_optimizer, cipt_loss, cfg)
        report["results"].append(item)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rows = summary_rows(report)
    print_summary(rows)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if output is None:
        print(encoded)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects against another audit writing here.
        with output.open("x", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        with output.with_suffix(".csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved JSON: {output}\nSaved CSV: {output.with_suffix('.csv')}", file=sys.stderr)


if __name__ == "__main__":
    main()
