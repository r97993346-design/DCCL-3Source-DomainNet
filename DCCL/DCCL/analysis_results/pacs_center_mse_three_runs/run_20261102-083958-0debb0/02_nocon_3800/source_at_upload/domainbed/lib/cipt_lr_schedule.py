"""Optional CIPTDCCL learning-rate schedule.

The upstream CIPT reproduction uses Adam with cosine annealing over 30 epochs.
DomainBed trains with a user-selected number of optimizer steps; the cosine
option here maps the same annealing *shape* to the full step budget, while the
original fixed-LR behavior remains the default.
Both the legacy CIPT/TDA objective and direct-E env_anchor objective are
supported; the schedule does not change their losses or model components.
"""
import math

import torch


def make_cipt_lr_scheduler(algorithm, hparams, total_steps):
    """Return None for fixed LR or a per-optimizer-step cosine scheduler.

    The scheduler must be stepped ONCE *after* each algorithm.update(), which
    performs optimizer.step() internally. total_steps is the actual loop length
    after train_all.py's checkpoint rounding (e.g., 3000 -> 3001).
    """
    mode = str(hparams.get("cipt_lr_schedule", "constant")).lower()
    if mode not in ("constant", "cosine"):
        raise ValueError("cipt_lr_schedule must be constant or cosine")
    if mode == "constant":
        return None

    objective = str(hparams.get("cipt_objective", "legacy")).lower()
    if objective not in ("legacy", "env_anchor"):
        raise ValueError(
            "cipt_lr_schedule=cosine currently supports CIPTDCCL legacy and env_anchor only"
        )
    if not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    optimizer = algorithm.optimizer
    if not optimizer.param_groups:
        raise ValueError("optimizer has no parameter groups")
    min_lr = float(hparams.get("cipt_lr_min", 0.0))
    initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if (not math.isfinite(min_lr) or min_lr < 0
            or any(not math.isfinite(lr) or lr <= 0 or min_lr >= lr for lr in initial_lrs)):
        raise ValueError("cipt_lr_min must be finite and 0 <= min_lr < each initial LR")

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=min_lr
    )
