# PICCL log-based PACS HPO ranges

This document was derived from `experiment_logs/PACS/piccl_optuna_evidence_20260713` by reading each run's `log.txt`, `results.jsonl`, and `source_manifest.txt`.  The Optuna objective must use only non-target `env*_out` values; target accuracy is report-only.

## Historical comparability

All four runs are PACS, seed 0 / trial_seed 0, ResNet-50, deterministic, lr 5e-5, weight_decay 0.0, and 5000-step full runs.  The historical logs are primarily target-env 0 runs, so they support choosing a conservative shared PACS search space but are not enough evidence for target-specific tuning.

| Run | Algorithm | Key PICCL configuration | Source out mean | Target out report-only | Interpretation |
|---|---|---|---:|---:|---|
| `260712_13-37-00_pacs_piccl_seed0` | PICCL | alpha=0.5, ccc=1.0, gt=replace, residual/gate not active in older hparams | 0.9660 | 0.8582 | Strong source score but target below best candidate; large unweighted CCC (~6.58). |
| `260712_13-48-25_pacs_dccl_seed0` | DCCL | no PICCL | 0.9649 | 0.8289 | Baseline for source-only comparison. |
| `260713_22-39-33_piccl_recommended_from_full_results_seed0_full` | PICCL | ccc=0.1, connectivity=0.01, residual_scale=0.02, gate_bias=-4 | 0.9655 | 0.7946 | PICCL perturbation is almost inactive: feature_delta_ratio 0.0007 and gate_mean 0.0245. |
| `260713_22-47-00_pacs_piccl_candidate_preserve_dccl_seed0` | PICCL | alpha=0.25, ccc=1.0, connectivity=0.5, residual_scale=0.25, gate_bias=-2.5, gt=keep | 0.9703 | 0.8826 | Best logged source and target report-only; feature perturbation is meaningful but finite (feature_delta_ratio 0.2827). |

## Search table

| 参数 | 各历史实验取值 | 对应性能 | 指标表现 | 建议下限 | 建议上限 | 搜索类型 | 依据 |
|---|---:|---:|---|---:|---:|---|---|
| lr | all 5e-5 | source 0.9649-0.9703 | no contrast evidence | 3e-5 | 8e-5 | log float | Keep near proven stable value only. |
| piccl_lr_multiplier | 0.25, 0.5 | 0.9655 vs 0.9703 | 0.25 gives piccl_grad_norm 0.0031, 0.5 gives 0.0226 | 0.25 | 0.75 | float | 0.25 too weak; allow moderate expansion beyond 0.5. |
| piccl_rank | 16 | all PICCL | no evidence | 16 | 16 | fixed | No logged contrast; keep fixed. |
| piccl_alpha_max | 0.5, 0.25 | 0.9660/0.9655 vs 0.9703 | 0.25 best with gate path | 0.15 | 0.35 | float | 0.5 may be excessive; search around candidate. |
| piccl_ccc_weight | 1.0, 0.1 | 0.9703 best at 1.0 | 0.1 weighted CCC 0.385 too weak; 1.0 weighted CCC 5.16 acceptable | 0.3 | 1.2 | float | This scales connectivity/int/ref CCC as a whole, not a tiny regularizer. |
| piccl_connectivity_weight | 0.01, 0.5 | 0.9655 vs 0.9703 | 0.01 makes CCC path weak | 0.1 | 0.7 | float | Avoid near-zero; allow around 0.5. |
| piccl_isr_weight | 0.1, 0.03, 0.05 | 0.9660/0.9655/0.9703 | weighted ISR 0.030 at 0.05 | 0.05 | 0.05 | fixed | Insufficient independent evidence; candidate value balanced. |
| piccl_inv_weight | 0.05, 0.01 | 0.9660 vs 0.9703 | weighted inv tiny at 0.01 | 0.01 | 0.01 | fixed | Low contribution and no evidence to expand. |
| piccl_orth_weight | 0.001, 0.0001 | 0.9660 vs 0.9703 | weighted orth negligible | 0.0001 | 0.0001 | fixed | Keep candidate. |
| piccl_residual_scale | 0.02, 0.25 | 0.9655 vs 0.9703 | 0.02 almost no effect; 0.25 meaningful | 0.08 | 0.30 | float | Avoid inactive perturbation; cap near observed finite perturbation. |
| piccl_gate_bias | -4, -2.5 | 0.9655 vs 0.9703 | gate_mean 0.0245 vs 0.339 | -3.5 | -2.0 | float | -4 closes gate; search around partially open gate. |
| piccl_delayed_start_ratio | 0.3, 0.1 | 0.9655 vs 0.9703 | 0.3 delays PICCL too long | 0.05 | 0.20 | float | Favor earlier candidate but keep warmup. |
| piccl_loss_warmup_ratio | 0.2, 0.25 | similar | no independent evidence | 0.25 | 0.25 | fixed | Keep best candidate. |
| piccl_feature_warmup_ratio | 0.2, 0.3 | 0.9655 vs 0.9703 | candidate higher but active | 0.20 | 0.40 | float | Search around 0.3. |
| piccl_fusion_mode | residual_gate | best candidate | current active mode | residual_gate | residual_gate | fixed | No structure redesign. |
| t | 0.1 | all | no evidence | 0.1 | 0.1 | fixed | DCCL base parameter held fixed. |

## Final Optuna parameters

Searched: `lr`, `piccl_lr_multiplier`, `piccl_alpha_max`, `piccl_ccc_weight`, `piccl_connectivity_weight`, `piccl_residual_scale`, `piccl_gate_bias`, `piccl_delayed_start_ratio`, `piccl_feature_warmup_ratio`.

Fixed: `piccl_rank=16`, `piccl_isr_weight=0.05`, `piccl_inv_weight=0.01`, `piccl_orth_weight=0.0001`, `piccl_loss_warmup_ratio=0.25`, `t=0.1`, and structural PICCL settings. These have insufficient independent evidence or are not part of this HPO task.
