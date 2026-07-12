# PICCL Regression Analysis and Leakage-Safe HPO Plan

## Evidence sources

Parsed in full, not just final rows:

- PICCL: `train_output/PACS/260712_13-37-00_pacs_piccl_seed0/log.txt` and `results.jsonl` (193 checkpoint records).
- DCCL: `train_output/PACS/260712_13-48-25_pacs_dccl_seed0/log.txt` and `results.jsonl` (204 checkpoint records).
- Code path: `train_all.py` -> `domainbed/trainer.py` -> `domainbed/algorithms/algorithms.py::DCCL` or `domainbed/algorithms/piccl.py::PICCL`.

## Quantitative comparison

| Item | PICCL | DCCL | Fairness conclusion |
|---|---:|---:|---|
| Dataset | PACS | PACS | Same |
| Environments | env0 art_painting, env1 cartoon, env2 photo, env3 sketch | same | Same |
| Test envs | `[[0], [1], [2], [3]]` leave-one-domain-out | same | Same |
| Seed / trial_seed | 0 / 0 | 0 / 0 | Same |
| Backbone | resnet50 | resnet50 | Same |
| Batch size | 32 per source domain, target batch 0; total 96 | same | Same |
| Steps | requested 5001, checkpoint freq 100 | same | Same |
| Optimizer | adam | adam | Same family |
| LR / WD | 5e-05 / 0.0 | 5e-05 / 0.0 | DCCL groups preserved; PICCL adds new groups |
| Scheduler | none found in logs/code path | none found | Same |
| Augmentation | `data_augmentation=True`, `aug=0`, `val_augment=False` | same | Same |
| DCCL weights | `l=1`, `l_d=0.01` effective hparam, `l_layer=1`, `t=0.1`, `t_pre=0.2` | same | Same |
| PICCL extra weights | ccc=1.0, int=0.1, ref=1.0, isr=0.1, orth=0.001, inv=0.05, alpha max=0.5 | n/a | Added variable |
| Params | 49,205,455 | 49,132,487 | PICCL adds 72,968 params |
| First-step mean loss scale | ~13.79 | ~12.20 | PICCL is higher before alpha becomes nonzero because CCC/ref/cross losses are active |
| Best mean target out | 89.03% | 89.47% | PICCL -0.44 pp |
| Final mean target out | 83.08% | 83.86% | PICCL -0.78 pp |

Per target-domain best/final target out accuracy:

| Target env | PICCL best step/out | PICCL final step/out | DCCL best step/out | DCCL final step/out | Best delta | Final delta |
|---|---:|---:|---:|---:|---:|---:|
| art_painting (0) | 900 / 0.8949 | 5000 / 0.8582 | 2300 / 0.9022 | 5000 / 0.8289 | -0.0073 | +0.0293 |
| cartoon (1) | 2600 / 0.8611 | 5000 / 0.8184 | 2500 / 0.8590 | 5000 / 0.8120 | +0.0021 | +0.0064 |
| photo (2) | 500 / 0.9671 | 5000 / 0.9551 | 500 / 0.9731 | 5000 / 0.9341 | -0.0060 | +0.0210 |
| sketch (3) | 2400 / 0.8382 | 3900 / 0.6917 | 3800 / 0.8446 | 5000 / 0.7796 | -0.0064 | -0.0879 |

## Direct diagnosis by required categories

| Category | Evidence | Conclusion | Confidence |
|---|---|---|---|
| A. unfair config mismatch | Logs show same PACS split, seed, backbone, batch size, optimizer, LR, WD, steps, augmentation, and DCCL weights. | Not the primary cause. | High |
| B. PICCL changed DCCL data flow | PICCL classifies and projects `m = causal_mediator(z, ...)`; original DCCL uses raw `feature_x`. | Yes: classifier/SupCon/pretrained-feature branch consume mediated features in legacy mode. | High |
| C. PICCL loss scale too large | Step 0 PICCL loss includes `loss_ccc≈12` while DCCL total is ≈12 including CE/SupCon/pre-cl; PICCL adds this on top of CE. | Likely contributor. | High |
| D. fusion strength too large | Legacy mediator is `LayerNorm(z - alpha*sensitive)` with alpha ramping to 0.5 and no gate/residual scale. | Likely contributor. | High |
| E. new module not updated | Optimizer includes `sensitive_subspace` and `causal_mediator`; new tests cover nonzero gradients for residual gate. | Not evident. | Medium |
| F. duplicate optimizer params | Audit found disjoint groups; regression test now asserts this. | Not evident. | High |
| G. DCCL params/LR changed | PICCL reuses DCCL group LRs for featurizer/classifier/projectors/encoders and adds groups. | Not evident for original groups. | High |
| H. normalization position | PICCL applies LayerNorm before classifier/projector; DCCL does not. | Possible important data-flow change. | High |
| I. detach/no_grad error | Reference encoder is no_grad/frozen as intended; basis task gradient is configurable and default detached. | No clear bug. | Medium |
| J. BatchNorm affected by extra forward | Trainable featurizer is forwarded on both `all_x` and `all_x_2` as DCCL already does; frozen pre_featurizer is forced eval. | Low likelihood. | Medium |
| K. masks/self-mask | `_cross_domain_supcon` excludes self diagonal and requires same label/different domain. | No diagonal bug found. | High |
| L. architecture unreasonable | Random low-rank subtraction plus LayerNorm replaces raw features for all task heads once enabled. | Needs safer mode. | High |
| M. normal training but untuned HPs | Best mean drop is modest (-0.44 pp), final drop concentrated on sketch; current PICCL defaults were not adapted. | Also likely. | Medium-high |

## Architecture audit

Current legacy flow:

`input -> featurizer(z, inter_feats) -> causal_mediator(z, sensitive_subspace, alpha) -> classifier(m)`.
The same mediated feature `m` also feeds `proj_head` for cross-domain SupCon and intervention NT-Xent, and `pre_proj_head(m)` for the pretrained-reference contrastive loss. The original feature `z` is retained only for response estimation and residual-bank updates, not as the direct classifier/projector input.

Key concerns:

1. Legacy PICCL effectively replaces DCCL features with `LayerNorm(z - alpha * sensitive_projection(z))`.
2. At step 0 alpha is 0, but `LayerNorm(z)` still changes the classifier input relative to DCCL raw `z`.
3. PICCL CCC/reference/cross losses are active at full weight from step 0, even when feature intervention alpha is zero.
4. The added residual-bank state is batch-history dependent but uses only source minibatches in the current leave-one-target-out training loop.
5. No target-domain metric is required for training, but previous manual interpretation could accidentally rank by `test_out`; the new HPO code prevents that.

## Implemented changes and motivation

- Preserved `piccl_fusion_mode=legacy` as the default so old commands keep old PICCL behavior.
- Added `piccl_fusion_mode=residual_gate`: `fused = original + residual_scale * warmup_alpha * sigmoid(gate(original)+gate_bias) * (piccl_feature-original)`.
- Zero-initialized gate weights and negative default `piccl_gate_bias=-4` keep the new path near-closed at startup.
- Added independent PICCL LR multiplier, delayed start, feature warmup, loss warmup, residual scale, gate bias, connectivity weight, and diagnostic metrics.
- Added `use_piccl=false` fallback to the inherited DCCL update/predict data flow for regression checks.
- Added HPO wrapper whose objective is exactly `mean(source_domain_out_acc) - 0.2 * std(source_domain_out_acc)` and excludes `real_test_envs`.

## Search space

Stage 1 random/successive-halving samples only stability parameters: PICCL total CCC weight, connectivity weight, residual scale, PICCL LR multiplier, delayed start, loss/feature warmup, temperature if present, and residual-gate bias. Structural choices such as hidden dimensions, projector depth, backbone, heads, and broad top-k/mask settings are intentionally excluded.

## Recommended commands

Dry run:

```bash
cd DCCL/DCCL
python scripts/tune_piccl.py --config configs/piccl_hpo.json --dry-run --trials 2
```

Stage 1 quick search:

```bash
cd DCCL/DCCL
CUDA_VISIBLE_DEVICES=0 python scripts/tune_piccl.py --config configs/piccl_hpo.json --output train_output/piccl_hpo/stage1 --trials 24 --seed 0 --gpu 0 --resume
```

Stage 2 full training of selected candidates: use `best_command.sh` generated by Stage 1, or override the config `budgets` to `[5001]` and restrict to the top five parameter sets.

Stage 3 multi-seed verification: rerun the Stage 2 top two parameter sets with `--trial_seed 0`, `--trial_seed 1`, and `--trial_seed 2`, reporting target-domain metrics only after parameter selection.

## Still not determined from existing logs

- Exact GPU model and peak memory were not logged.
- No per-source validation objective was recorded at training time; it can be reconstructed from env out splits per target run.
- No gradient norms or feature-delta diagnostics existed in the original logs; these are now added for future runs.
- The full HPO experiments are intentionally not run in this code change because they are expensive.
