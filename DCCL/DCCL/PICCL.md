# PICCL: Paired-Intervention Causal Connectivity Learning

PICCL is an independent algorithm registered through `--algorithm PICCL`. It
reuses the DCCL training infrastructure but replaces the trainable connection
representation with a causal mediator obtained from paired intervention
responses.

## Data flow

1. `x` and the existing strong view `x_2` are encoded by the trainable encoder.
2. The same pair is encoded by the frozen pre-trained reference encoder.
3. PIRE computes
   `delta = (z_int - z) - stopgrad(z0_int - z0)`.
4. CDRM maintains class/domain EMA residual prototypes.
5. ISSL learns a low-rank intervention-sensitive basis from detached PIRE and
   CDRM responses.
6. CMP computes `m = LayerNorm(z - alpha * Proj_sensitive(z))`.
7. CCC combines intervention, same-class cross-domain and reference-preserving
   connections. The classifier reads `m`, not the raw adapted feature.

The frozen reference encoder is an intervention-response control and is not a
knowledge-distillation teacher.

## Main files

- `domainbed/algorithms/piccl.py`: PICCL and its modules.
- `domainbed/algorithms/__init__.py`: algorithm registration.
- `domainbed/hparams_registry.py`: PICCL hyperparameters.
- `configs/piccl_domainnet.yaml`: recommended DomainNet defaults.
- `tests/test_piccl.py`: core unit tests.

## Three-source DomainNet example

Run from `DCCL/DCCL` and replace the data path and environment ids with the
existing validated split:

```bash
python -u train_all.py piccl_domainnet \
  configs/piccl_domainnet.yaml \
  --data_dir /path/to/data \
  --dataset DomainNet \
  --algorithm PICCL \
  --model resnet50 \
  --source_envs 1 2 3 \
  --target_env 4 \
  --steps 15000 \
  --seed 0 \
  --trial_seed 0
```

For a short integration check:

```bash
python -u train_all.py piccl_smoke \
  configs/piccl_domainnet.yaml \
  --data_dir /path/to/data \
  --dataset DomainNet \
  --algorithm PICCL \
  --model resnet50 \
  --source_envs 1 2 3 \
  --target_env 4 \
  --steps 10 \
  --checkpoint_freq 5 \
  --debug
```

For a short run, override `piccl_total_steps=10` through the repository's
existing `sconf` CLI override syntax so that alpha/int-connectivity ramp within
the short schedule.

## GT ablation

- `piccl_gt_mode: replace` (default): replaces DCCL GT/VAE with PICCL.
- `piccl_gt_mode: keep`: adds the original GT/VAE objective for an ablation.

`--algorithm DCCL` does not instantiate or read PICCL modules.
