"""Audit Adam and cosine call order using scalar learning rates only."""

import json
import math
import os
import runpy
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook

from protocol import CODE, cosine_lr, git, read_json, require


class CosineTrace:
    def __init__(self, initial, updates):
        self.initial = initial
        self.updates = updates
        self.optimizer = None
        self.scheduler_steps = 0
        self.optimizer_steps = 0
        self.rows = []

    def pre(self, optimizer, args, kwargs):
        require(isinstance(optimizer, torch.optim.Adam) and len(optimizer.param_groups) == 1, "expected one Adam group")
        if self.optimizer is None:
            self.optimizer = optimizer
        require(optimizer is self.optimizer and self.optimizer_steps == self.scheduler_steps, "incorrect Adam call order")
        index = self.optimizer_steps
        lr = optimizer.param_groups[0]["lr"]
        require(lr > 0.0 and math.isclose(lr, cosine_lr(index, self.initial, self.updates),
                                         rel_tol=1e-7, abs_tol=1e-16), f"invalid LR at step {index}")
        self.rows.append({"step": index, "lr_used": lr})

    def post(self, optimizer, args, kwargs):
        self.optimizer_steps += 1

    def before_scheduler(self, scheduler):
        require(scheduler.optimizer is self.optimizer and scheduler.T_max == self.updates
                and scheduler.eta_min == 0 and self.optimizer_steps == self.scheduler_steps + 1,
                "expected one cosine call after each Adam update")

    def after_scheduler(self, scheduler):
        self.scheduler_steps += 1
        lr = scheduler.optimizer.param_groups[0]["lr"]
        require(scheduler.last_epoch == self.scheduler_steps
                and math.isclose(lr, cosine_lr(self.scheduler_steps, self.initial, self.updates),
                                 rel_tol=1e-7, abs_tol=1e-16), "incorrect cosine LR")
        self.rows[-1]["lr_after_scheduler"] = lr


def main():
    destination = Path(sys.argv[1])
    args = sys.argv[2:]
    config = read_json(Path(args[1]))
    updates = int(args[args.index("--steps") + 1])
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    require(torch.cuda.device_count() == 1 and f"GPU-{torch.cuda.get_device_properties(0).uuid}".lower()
            == visible.lower(), "training process is not bound to its assigned GPU")
    print(f"CUDA_PHYSICAL_UUID {visible}", flush=True)
    print(f"TRAINING_CODE_COMMIT {git('rev-parse', 'HEAD')}", flush=True)
    sys.path.insert(0, str(CODE))
    from domainbed.algorithms.cipt_prompt import PromptLearner

    trace = CosineTrace(config["lr"], updates)
    original_init = PromptLearner.__init__
    original_step = CosineAnnealingLR.step
    observed = []

    def checked_init(self, *positional, **named):
        original_init(self, *positional, **named)
        length = self.context.shape[0]
        require(length == config["cipt_prompt_length"], "effective prompt length differs")
        observed.append(length)
        print(f"PROMPT_CONTEXT_LENGTH {length}", flush=True)

    def checked_step(self, *positional, **named):
        count = self.last_epoch >= 0  # Exclude PyTorch's constructor initialization at t=0.
        if count:
            trace.before_scheduler(self)
        output = original_step(self, *positional, **named)
        if count:
            trace.after_scheduler(self)
        return output

    first = register_optimizer_step_pre_hook(trace.pre)
    second = register_optimizer_step_post_hook(trace.post)
    PromptLearner.__init__ = checked_init
    CosineAnnealingLR.step = checked_step
    entry = CODE / "train_all.py"
    sys.argv = [str(entry), *args]
    try:
        runpy.run_path(str(entry), run_name="__main__")
        require(observed and trace.optimizer_steps == trace.scheduler_steps == updates
                and trace.rows[-1]["lr_after_scheduler"] == 0.0, "incomplete Adam or cosine schedule")
        temporary = destination.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in trace.rows))
        temporary.replace(destination)
        print(f"ADAM_UPDATES {trace.optimizer_steps} COSINE_UPDATES {trace.scheduler_steps}", flush=True)
    finally:
        first.remove()
        second.remove()
        PromptLearner.__init__ = original_init
        CosineAnnealingLR.step = original_step


if __name__ == "__main__":
    main()
