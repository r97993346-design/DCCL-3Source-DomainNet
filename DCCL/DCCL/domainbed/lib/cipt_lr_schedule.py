"""Optional CIPTDCCL learning-rate schedule.

The upstream CIPT reproduction uses Adam with cosine annealing over 30 epochs.
The default cosine option maps that shape to the configured optimizer-step
budget. The opt-in mixed-source strategy instead supports one scheduler update
per complete epoch. Original fixed-LR behavior remains the default.
Both the legacy CIPT/TDA objective and direct-E env_anchor objective are
supported; the schedule does not change their losses or model components.
"""
import math

import torch


def make_cipt_lr_scheduler(algorithm, hparams, total_steps, *, total_epochs=None):
    """Return None for fixed LR or a cosine scheduler with an explicit unit.

    With cipt_lr_step_unit=step (default), advance after each optimizer update.
    With cipt_lr_step_unit=epoch, advance only after the last batch of an epoch,
    using total_epochs as T_max. The trainer owns this boundary decision.
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
    unit = str(hparams.get("cipt_lr_step_unit", "step")).lower()
    if unit not in ("step", "epoch"):
        raise ValueError("cipt_lr_step_unit must be step or epoch")
    period = total_steps
    if unit == "epoch":
        if (not isinstance(total_epochs, int) or isinstance(total_epochs, bool)
                or total_epochs < 1):
            raise ValueError("epoch LR stepping requires a positive total_epochs")
        period = total_epochs
    optimizer = algorithm.optimizer
    if not optimizer.param_groups:
        raise ValueError("optimizer has no parameter groups")
    min_lr = float(hparams.get("cipt_lr_min", 0.0))
    initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if (not math.isfinite(min_lr) or min_lr < 0
            or any(not math.isfinite(lr) or lr <= 0 or min_lr >= lr for lr in initial_lrs)):
        raise ValueError("cipt_lr_min must be finite and 0 <= min_lr < each initial LR")

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=period, eta_min=min_lr
    )
