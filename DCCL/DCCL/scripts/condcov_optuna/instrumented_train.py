"""Observe prompt construction and actual Adam updates, then run train_all."""

import functools
import json
import math
import os
import runpy
import sys
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

from protocol import CODE, DIAG_STEPS, STEPS, require


class AdamTrace:
    def __init__(self, destination):
        self.destination = destination
        self.step = 0
        self.before = None
        self.samples = []

    def pre(self, optimizer, args, kwargs):
        require(isinstance(optimizer, torch.optim.Adam), "expected Adam")
        if self.step in DIAG_STEPS:
            parameters = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
            self.before = [(p, p.detach().clone()) for p in parameters]

    def post(self, optimizer, args, kwargs):
        if self.before is not None:
            with torch.no_grad():
                gradients = [p.grad.detach().float().norm() for p, _ in self.before]
                changes = [(p.detach().float() - old.float()).norm() for p, old in self.before]
                parameters = [old.float().norm() for _, old in self.before]
                grad, change, parameters_norm = torch.stack([
                    torch.stack(values).norm() for values in (gradients, changes, parameters)
                ]).tolist()
            values = {
                "step": self.step, "grad_norm": grad, "update_norm": change, "param_norm": parameters_norm,
                "relative_update": change / max(parameters_norm, 1e-30),
                "lr": [g["lr"] for g in optimizer.param_groups],
            }
            require(all(math.isfinite(v) for v in (grad, change, parameters_norm)),
                    "non-finite optimizer telemetry")
            with self.destination.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(values) + "\n")
            self.samples.append(self.step)
            self.before = None
        self.step += 1


def main():
    path = Path(sys.argv[1])
    expected = int(sys.argv[2])
    training_args = sys.argv[3:]
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    require(torch.cuda.device_count() == 1
            and f"GPU-{torch.cuda.get_device_properties(0).uuid}".lower() == visible.lower(),
            "training process is not bound to its assigned physical GPU")
    print(f"CUDA_PHYSICAL_UUID {visible}", flush=True)
    sys.path.insert(0, str(CODE))
    from domainbed.algorithms.cipt_prompt import PromptLearner

    original_init = PromptLearner.__init__
    observed = []

    @functools.wraps(original_init)
    def checked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        length = self.context.shape[0]
        require(length == expected, f"prompt context has {length} tokens, expected {expected}")
        observed.append(length)
        print(f"PROMPT_CONTEXT_LENGTH {length}", flush=True)

    trace = AdamTrace(path)
    pre = register_optimizer_step_pre_hook(trace.pre)
    post = register_optimizer_step_post_hook(trace.post)
    PromptLearner.__init__ = checked_init
    entry = CODE / "train_all.py"
    sys.argv = [str(entry), *training_args]
    try:
        runpy.run_path(str(entry), run_name="__main__")
        require(observed and trace.step == STEPS and trace.samples == DIAG_STEPS,
                f"incomplete telemetry: prompt {observed}, updates {trace.step}, samples {trace.samples}")
    finally:
        PromptLearner.__init__ = original_init
        pre.remove()
        post.remove()


if __name__ == "__main__":
    main()
