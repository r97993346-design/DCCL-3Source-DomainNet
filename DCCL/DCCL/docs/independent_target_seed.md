# Independent random seeds for target-domain training tasks

Each call to `domainbed.trainer.train()` now resets Python, NumPy and PyTorch
random generators to `args.seed` before dataset, DataLoader and model setup.
`torch.manual_seed()` seeds both CPU and CUDA generators. Every target uses the
same explicit `--seed` value; no seed is derived from task order or step budget.
The reset also applies to separately invoked algorithm/baseline training tasks.

`--trial_seed` continues to control dataset splits. Changing `--seed` does not
change those split seeds. Model, loss, optimizer, learning-rate schedule, metrics
and output-path configuration are otherwise unchanged.

This isolates a later target from randomness consumed by earlier targets. For
example, training Art for a different number of steps no longer changes Sketch's
initial RNG state when both are run by the same `train_all.py` process. Startup
logs record `Reset training RNG: test_envs=..., seed=..., trial_seed=...`.

Use the same training seed and trial seed when comparing configurations for one
target. With fixed LR, matching evaluation frequency and deterministic execution,
different training budgets can share the same prefix. Cosine schedules spanning
different budgets still have different LR trajectories. Different hardware,
library versions, worker settings or nondeterministic kernels can also affect
reproducibility. Saved historical results retain their original execution policy.

Regression check (CPU-only synthetic datasets; no CLIP weights required):

```bash
python -m unittest discover -s tests -p 'test_target_seed_isolation.py' -v
```

The tests execute the real training loop, random sampler and evaluator, checking
independence from prior-task budgets and ordering, fixed-LR prefix matching, and
the separation of training seeds from split seeds.
