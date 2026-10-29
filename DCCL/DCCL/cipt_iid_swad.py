"""Source-only IID selection and dense LossValley SWAD for frozen-CLIP CIPT.

The existing DomainBed LossValley and AveragedModel implementations are reused.
Only trainable CIPT weights enter their queues: frozen CLIP is never copied,
averaged, moved to CPU, or updated by model selection.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import logging
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from domainbed.lib.swa_utils import AveragedModel
from domainbed.swad import LossValley


@contextmanager
def preserve_rng():
    """Evaluation must not change the next minibatch/template random draws."""
    py, np_state = random.getstate(), np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@torch.no_grad()
def trainable_state(model):
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters() if p.requires_grad
    }


@torch.no_grad()
def load_trainable_state(model, state):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if params.keys() != state.keys():
        raise ValueError("Checkpoint trainable parameter names differ from model")
    for name, param in params.items():
        if param.shape != state[name].shape:
            raise ValueError(f"Checkpoint shape mismatch for {name}")
        param.copy_(state[name].to(device=param.device, dtype=param.dtype))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, state, metadata):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"trainable_state_dict": state, "metadata": metadata}, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate_loader(model, loader, device):
    """Use exactly the TDA inference logits for both accuracy and val CE."""
    was_training = model.training
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    try:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model.predict(images, use_tda=True)
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.numel()
    finally:
        model.train(was_training)
    if total == 0:
        raise ValueError("Cannot evaluate an empty partition")
    return {"accuracy": correct / total, "loss": loss_sum / total, "num_samples": total}


def evaluate_sources(model, source_loaders, device):
    """Match DomainBed train_out/tr_outloss: equal weight per source domain."""
    if not source_loaders:
        raise ValueError("Source validation loaders are required")
    with preserve_rng():
        per_domain = {
            name: evaluate_loader(model, loader, device)
            for name, loader in source_loaders.items()
        }
    count = len(per_domain)
    total = sum(v["num_samples"] for v in per_domain.values())
    result = {
        "accuracy": sum(v["accuracy"] for v in per_domain.values()) / count,
        "loss": sum(v["loss"] for v in per_domain.values()) / count,
        "pooled_accuracy": sum(v["accuracy"] * v["num_samples"] for v in per_domain.values()) / total,
        "per_domain": per_domain,
    }
    if not all(math.isfinite(result[key]) for key in ("accuracy", "loss")):
        raise ValueError("Non-finite source validation metric")
    return result


class IIDSelector:
    """Keep the earliest checkpoint with maximal source-domain macro accuracy."""

    def __init__(self):
        self.state = None
        self.metadata = None

    def consider(self, model, validation, epoch, step):
        accuracy = float(validation["accuracy"])
        if not math.isfinite(accuracy):
            raise ValueError("Non-finite source validation accuracy")
        if self.metadata is not None and accuracy <= self.metadata["source_val"]:
            return False
        self.state = trainable_state(model)
        self.metadata = {
            "selection": "iid", "epoch": epoch, "step": step,
            "source_val": accuracy, "source_val_loss": validation["loss"],
            "source_val_per_domain": validation["per_domain"],
        }
        return True


class TrainableWeights(nn.Module):
    """Small module accepted by DomainBed AveragedModel, with no CLIP reference."""

    def __init__(self, model):
        super().__init__()
        items = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not items:
            raise ValueError("SWAD requires trainable parameters")
        self.names = tuple(name for name, _ in items)
        self.values = nn.ParameterList(p for _, p in items)


class DeviceLossValley(LossValley):
    """Original LossValley finalization with .to(device) instead of .cuda()."""

    def __init__(self, *args, device, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def get_final_model(self):
        if not self.converge_Q:
            raise RuntimeError("SWAD received no validation checkpoints")
        if not self.is_converged:
            self.evaluator.logger.warning(
                "SWAD valley did not converge; returning the last dense segment. "
                "See swad.status in metrics.json."
            )
            return self.converge_Q[-1].to(self.device)
        if not self.dead_valley:
            self.smooth_Q.popleft()
            while self.smooth_Q:
                if self.get_smooth_loss(0) > self.threshold:
                    break
                segment = self.smooth_Q.popleft()
                self.final_model.update_parameters(segment, step=segment.end_step)
        return self.final_model.to(self.device)


class SWADTracker:
    def __init__(self, model, device, n_converge=3, n_tolerance=6, tolerance_ratio=0.3):
        if n_converge < 2 or n_tolerance < 2 or tolerance_ratio < 0:
            raise ValueError("SWAD windows must be >= 2 and tolerance ratio >= 0")
        self.live = TrainableWeights(model)
        self.device = device
        self.segment = AveragedModel(self.live)
        self.valley = DeviceLossValley(
            SimpleNamespace(logger=logging.getLogger("cipt.swad")), device=device,
            n_converge=n_converge, n_tolerance=n_tolerance,
            tolerance_ratio=tolerance_ratio,
        )
        self.finished = False

    def update(self, step):
        if self.finished:
            raise RuntimeError("SWAD was already finalized")
        if not self.valley.dead_valley:
            # Called after EVERY optimizer step, not just validation epochs.
            self.segment.update_parameters(self.live, step=step)

    def checkpoint(self, validation):
        if self.valley.dead_valley:
            return
        if int(self.segment.n_averaged) == 0:
            raise RuntimeError("Cannot submit an empty SWAD segment")
        self.valley.update_and_evaluate(
            self.segment, validation["accuracy"], validation["loss"], lambda *_: None,
        )
        if not self.valley.dead_valley:
            self.segment = AveragedModel(self.live)

    def finalize(self):
        if self.finished:
            raise RuntimeError("SWAD can only be finalized once")
        self.finished = True
        averaged = self.valley.get_final_model()
        state = {
            name: value.detach().cpu().clone()
            for name, value in zip(averaged.module.names, averaged.module.values)
        }
        return state, {
            "status": "loss_valley" if self.valley.is_converged else "not_converged_last_segment",
            "start_step": averaged.start_step, "end_step": averaged.end_step,
            "n_averaged": int(averaged.n_averaged),
            "converge_step": self.valley.converge_step,
            "dead_valley": self.valley.dead_valley,
            "loss_threshold": None if self.valley.threshold is None else float(self.valley.threshold),
        }


def result_table(results, metric):
    """Fraction-valued rows; presentation formats percentages only once."""
    domains = [r["target"] for r in results]
    columns = ["Selection", *domains, "Avg."]
    rows = []
    for selection in ("iid", "last", "SWAD"):
        if not all(selection in r["selections"] for r in results):
            continue
        values = [r["selections"][selection][metric] for r in results]
        rows.append([selection, *values, sum(values) / len(values)])
    return columns, rows


def table_markdown(columns, rows):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join([row[0], *[f"{v:.3%}" for v in row[1:]]]) + " |" for row in rows)
    return "\n".join(lines)


def write_summary(directory, dataset, variant, results, seed, trial_seed):
    if not results:
        raise ValueError("No target results to summarize")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ids = "-".join(str(r["test_env"]) for r in results)
    stem = f"summary_targets_{ids}_seed_{seed}_trial_{trial_seed}"
    tables, sections = {}, []
    for metric in ("target_out", "target_in", "source_val"):
        columns, rows = result_table(results, metric)
        tables[metric] = {"columns": columns, "rows": rows}
        sections.append(f"{metric} (percent; source validation selects models)\n\n" + table_markdown(columns, rows))
        with (directory / f"{stem}_{metric}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows([[row[0], *[f"{v:.6%}" for v in row[1:]]] for row in rows])
    summary = {
        "schema_version": 1, "dataset": dataset, "model_variant": variant,
        "seed": seed, "trial_seed": trial_seed, "primary_target_metric": "target_out",
        "targets": results, "tables": tables,
    }
    write_json(directory / f"{stem}.json", summary)
    (directory / f"{stem}.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    columns, rows = result_table(results, "target_out")
    print(f"\n{dataset} target_out\n" + table_markdown(columns, rows), flush=True)
    print(f"Summary: {directory / (stem + '.md')}", flush=True)
    return summary
