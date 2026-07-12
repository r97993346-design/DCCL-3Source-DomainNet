# PICCL Stage 0 Bypass Regression Analysis

## Scope

This report covers only the Stage 0 bypass mismatch between original `DCCL` and `PICCL` with `use_piccl=false`. HPO, `residual_scale=0`, and all-PICCL-loss-weight-zero experiments remain paused until the bypass path is identical.

## Experiments read

| Run | Files | Records read |
| --- | --- | ---: |
| Original DCCL | `train_output/PACS/260712_17-51-30_stage0_dccl_seed0/log.txt`, `results.jsonl` | 164 log lines; 17 JSONL records |
| PICCL off | `train_output/PACS/260712_17-52-28_stage0_piccl_off_seed0/log.txt`, `results.jsonl` | 190 log lines; 16 JSONL records |

The comparison used the complete logs and every JSONL record, not only the final line.

## Fairness/configuration comparison

| Item | DCCL | PICCL with `use_piccl=false` | Assessment |
| --- | --- | --- | --- |
| Full command | `train_all.py stage0_dccl_seed0 --dataset PACS --algorithm DCCL --model resnet50 --deterministic --trial_seed 0 --checkpoint_freq 100 --steps 500 --data_dir /home/hooasia/lgg/data/repro_dccl_data` | `train_all.py stage0_piccl_off_seed0 --dataset PACS --algorithm PICCL --model resnet50 --deterministic --trial_seed 0 --checkpoint_freq 100 --steps 500 --data_dir /home/hooasia/lgg/data/repro_dccl_data --use_piccl false` | Intended algorithm switch only |
| dataset | PACS | PACS | identical |
| test_envs | `[[0], [1], [2], [3]]` | `[[0], [1], [2], [3]]` | identical target sweep |
| seed | 0 | 0 | identical |
| trial_seed | 0 | 0 | identical |
| hparams_seed | not present in either command/results | not present in either command/results | identical absence |
| backbone | `resnet50`, ImageNet pretrained | `resnet50`, ImageNet pretrained | identical |
| batch_size | 32 | 32 | identical |
| steps | 500 requested, 501 internal checkpoint loop | 500 requested, 501 internal checkpoint loop | identical |
| checkpoint_freq | 100 | 100 | identical |
| learning_rate | 5e-05 | 5e-05 | identical |
| weight_decay | 0.0 | 0.0 | identical |
| optimizer | adam | adam | identical optimizer type |
| scheduler | no scheduler shown/constructed in these code paths | no scheduler shown/constructed in these code paths | identical absence |
| data augmentation | `data_augmentation=True`, `aug=0`, `mix=0`, `val_augment=False` | same | identical |
| DCCL loss weights | effective hparams `l=1`, `l_d=0.01`, `l_layer=1` | same effective DCCL hparams | identical |
| data split | PACS env sizes: art 2048, cartoon 2344, photo 1670, sketch 3929; per-target train envs are complement of test env | same | identical |
| holdout_fraction | 0.2 | 0.2 | identical |
| class_balanced | False | False | identical |
| num_workers | not logged directly; DataLoader uses `dataset.N_WORKERS` | same code and dataset | no logged difference |
| deterministic/CUDNN/AMP | `--deterministic`; CUDNN deterministic true/benchmark false by code; AMP not used | same | identical |
| checkpoint selection | SWAD `LossValley`, `n_converge=3`, `n_tolerance=6`, `tolerance_ratio=0.3`; oracle/iid/last/SWAD reported | same | identical method |
| best/final step | checkpoint records at 0/100/200/300/400/500 for completed target envs; DCCL JSONL has 17 records, PICCL-off has 16 records in provided files | same schedule, provided files stop at different points | incomplete artifact count can affect aggregate comparison |
| per-env in/out acc | see table below | see table below | diverges after step 0 in some target folds |
| model parameter count | 49,132,487 | 53,365,703 before fix | **not identical; sufficient to explain divergence risk** |
| optimizer groups | 6 original DCCL groups by code | 9 groups before fix because PICCL groups were constructed even when off | **not identical before fix** |

## Accuracy/loss differences from all JSONL records

Per shared target/step checkpoint, the first target env step 0 was identical, but later checkpoints diverged. Examples:

| Target env | Step | DCCL out acc | PICCL-off out acc | Delta | DCCL loss | PICCL-off loss | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| art_painting | 0 | 0.178484 | 0.178484 | 0.000000 | 12.331758 | 12.331758 | 0.000000 |
| art_painting | 100 | 0.858191 | 0.789731 | -0.068460 | 8.762183 | 8.776302 | +0.014119 |
| art_painting | 500 | 0.870416 | 0.865526 | -0.004890 | 7.540774 | 7.532225 | -0.008549 |
| cartoon | 0 | 0.175214 | 0.151709 | -0.023505 | 12.286621 | 12.250942 | -0.035679 |
| cartoon | 500 | 0.831197 | 0.771368 | -0.059829 | 7.483642 | 7.506842 | +0.023200 |
| photo | 100 | 0.964072 | 0.952096 | -0.011976 | 9.120404 | 9.167239 | +0.046835 |
| photo | 300 | 0.961078 | 0.955090 | -0.005988 | 7.993214 | 7.990617 | -0.002597 |

Because the PICCL-off run has a different parameter count and extra optimizer groups, these performance differences should not be attributed to PICCL's intended architecture or loss behavior.

## Code-path trace and first mismatch

Path: CLI `--use_piccl false` -> `Config.argv_update()` -> `hparams["use_piccl"]` -> `algorithms.get_algorithm_class("PICCL")` -> `PICCL.__init__()` -> inherited `DCCL.__init__()` -> PICCL modules -> PICCL optimizer groups -> `PICCL.update()` bypasses to `super().update()` only after construction.

Before this fix, the first deterministic mismatch was at model construction, before any forward pass:

1. `PICCL.__init__()` always called `DCCL.__init__()` and then always constructed `sensitive_subspace`, `causal_mediator`, `residual_gate`, `residual_bank`, `pire`, and `piccl_alpha`.
2. The logged parameter count changed from 49,132,487 to 53,365,703.
3. The optimizer changed from the original six DCCL parameter groups to nine PICCL groups.
4. `update()` and `predict_embed()` used `bool(self.hparams.get("use_piccl", True))`, so a raw string value such as `"false"` would be truthy if it reached those branches unconverted.

The observed run did record `use_piccl: False` in JSONL, so its training forward path bypassed PICCL at update time. However, construction and optimizer setup had already diverged. The first mismatch is therefore **extra PICCL parameter registration and optimizer construction during `PICCL.__init__()`**.

## Root cause

`use_piccl=false` was implemented as a late runtime bypass around `update()`/`predict_embed()` rather than a construction-time DCCL bypass. Extra PICCL modules were still instantiated and the optimizer was replaced with PICCL parameter groups. This violates the Stage 0 requirement that bypass mode be the original DCCL path.

## Fix

- Added a strict `parse_bool()` helper so `false`, `False`, `0`, `off`, and `no` cannot be treated as truthy.
- `PICCL.__init__()` now records `self.use_piccl` immediately after inherited DCCL construction and returns before any PICCL-specific module is built when disabled.
- Disabled mode keeps the original `DCCL` optimizer from `DCCL.__init__()` and does not add PICCL parameter groups.
- Disabled mode keeps `predict_embed()` as raw DCCL `featurizer(x)` and `update()` as inherited `DCCL.update()`.
- `get_forward_model()` now returns the original DCCL `ForwardModel` in disabled mode.

## Data flow after fix

### Before fix (`use_piccl=false`)

`DCCL.__init__()` -> create DCCL modules and optimizer -> create PICCL modules -> replace optimizer with PICCL grouped optimizer -> `update()` delegates to DCCL.

### After fix (`use_piccl=false`)

`DCCL.__init__()` -> create DCCL modules and original DCCL optimizer -> parse disabled flag -> return. No PICCL module construction, no PICCL forward, no PICCL loss, no PICCL optimizer params, no scheduler/optimizer-group change.

## Regression coverage

`tests/test_piccl_regression.py` now verifies disabled bypass behavior with fixed CPU tensors and `torch.testing.assert_close`:

1. `use_piccl=false` main path matches original DCCL.
2. Fixed-input backbone features match.
3. Logits match.
4. DCCL `ce_loss`, `sup_cl_loss`, `pre_cl_loss`, and total loss match.
5. Optimizer parameter names and group counts match.
6. No PICCL-only parameters exist in disabled mode.
7. First update leaves corresponding parameters identical.
8. String/boolean/integer false values parse as disabled.

`scripts/compare_dccl_piccl_bypass.py` provides a standalone one-batch diagnostic that reports the first differing initialization parameter, feature, logits, loss, gradient/step-derived parameter, and exits non-zero on failure. It does not run full PACS training.

## Continue/stop decision

Stage 0 parameter search should remain paused until the fixed branch reruns the original Stage 0 commands and confirms identical bypass metrics. The code-level bypass invariant is now covered, but the provided full PACS Stage 0 artifacts predate this fix.

## Not yet verified

- Full PACS Stage 0 rerun after this patch.
- GPU/CUDA full-training determinism after this patch.
- PICCL enabled-mode performance after this bypass-only patch.

## Exact next Stage 0 rerun commands

```bash
python train_all.py stage0_dccl_seed0 --dataset PACS --algorithm DCCL --model resnet50 --deterministic --trial_seed 0 --checkpoint_freq 100 --steps 500 --data_dir /home/hooasia/lgg/data/repro_dccl_data
python train_all.py stage0_piccl_off_seed0 --dataset PACS --algorithm PICCL --model resnet50 --deterministic --trial_seed 0 --checkpoint_freq 100 --steps 500 --data_dir /home/hooasia/lgg/data/repro_dccl_data --use_piccl false
```
