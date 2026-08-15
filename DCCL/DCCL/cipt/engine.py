"""Small training and evaluation loops for CIPT."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor, nn

from .losses import cipt_loss
from .model import CIPTModel


def _move_batch(batch, device: torch.device) -> tuple[Tensor, Tensor]:
    if isinstance(batch, dict):
        images, labels = batch["image"], batch["label"]
    else:
        images, labels = batch[:2]
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def make_optimizer(model: CIPTModel, lr: float = 2.5e-3, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """Adam optimizer over prompt tokens, adapters, and TDA parameters."""

    return torch.optim.Adam(model.trainable_parameters(), lr=lr, weight_decay=weight_decay)


def train_one_epoch(
    model: CIPTModel,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    beta: float = 2.0,
    gamma: float = 5.0,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """Train CIPT for one epoch."""

    device = torch.device(device)
    model.train()
    meters: dict[str, list[float]] = defaultdict(list)

    for batch in dataloader:
        images, labels = _move_batch(batch, device)
        output = model(images, labels)
        if output.interventional_logits is None:
            raise RuntimeError("interventional_logits is None; pass labels when training with class-conditioned templates.")

        losses = cipt_loss(
            output.interventional_logits,
            output.causal_logits,
            output.spurious_logits,
            output.causal_features,
            output.spurious_features,
            labels,
            beta=beta,
            gamma=gamma,
        )

        optimizer.zero_grad(set_to_none=True)
        losses.loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.trainable_parameters(), max_grad_norm)
        optimizer.step()

        batch_size = images.shape[0]
        meters["loss"].append(float(losses.loss.detach()) * batch_size)
        meters["classification"].append(float(losses.classification.detach()) * batch_size)
        meters["decomposition"].append(float(losses.decomposition.detach()) * batch_size)
        meters["independence"].append(float(losses.independence.detach()) * batch_size)
        meters["causal_ce"].append(float(losses.causal_ce.detach()) * batch_size)
        meters["spurious_kl"].append(float(losses.spurious_kl.detach()) * batch_size)
        meters["num_samples"].append(batch_size)

    total = max(1, int(sum(meters["num_samples"])))
    return {key: sum(value) / total for key, value in meters.items() if key != "num_samples"}


@torch.no_grad()
def evaluate(model: CIPTModel, dataloader: Iterable, device: str | torch.device, use_tda: bool = True) -> dict[str, float]:
    """Evaluate top-1 accuracy."""

    device = torch.device(device)
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        images, labels = _move_batch(batch, device)
        logits = model.predict(images, use_tda=use_tda)
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return {"accuracy": correct / max(1, total), "num_samples": float(total)}


def cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int = 30,
    min_lr: float = 0.0,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine decay scheduler matching the paper's training recipe."""

    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

