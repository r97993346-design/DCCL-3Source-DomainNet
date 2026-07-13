# PICCL Stage-0 regression analysis

## Read inputs

This report is based on the full `log.txt` and every JSONL row from:

- `train_output/PACS/260712_18-26-22_stage0_dccl_seed0/`
- `train_output/PACS/260712_19-32-44_stage0_piccl_scale0_seed0/`

## Conclusion

The new residual-scale-0 experiment is **not a strict DCCL bypass**.  It used
`algorithm=PICCL`, `use_piccl=true`, `piccl_fusion_mode=residual_gate`, and
`piccl_residual_scale=0`, so the classification feature was fused as
`fused_feature = original_feature`.  However, PICCL still constructed and ran
its mediator, residual bank, PIRE path, and auxiliary diagnostics/losses.  The
logged loss weights in this run were `piccl_ccc_weight=0`, `piccl_isr_weight=0`,
`piccl_orth_weight=0`, and `piccl_inv_weight=0`, so those auxiliary losses were
computed but not added to `total_loss`.  The run still differs from DCCL because
PICCL replaces the DCCL update objective with classification loss only in this
configuration: `loss = loss_cls + weighted_piccl_losses`, while DCCL trains with
`CE + l * SupCon + l_layer * pre_CL + l_d * GT`.  Therefore the mismatch is an
expected semantic mismatch between **scale=0** and **strict DCCL bypass**, not a
failure of the residual gate to make fused features equal to original features.

## Residual-scale semantics from code

| Question | Answer |
| --- | --- |
| Does `residual_scale=0` only close feature fusion? | Yes for `piccl_fusion_mode=residual_gate`: the gate computes `original + scale * alpha * gate * (piccl-original)`, so `scale=0` gives the original feature. |
| Does the PICCL module still execute forward/update? | Yes unless `use_piccl=false` or new `piccl_strict_bypass=true`. |
| Is connectivity loss still computed? | Yes, `loss_cross`, `loss_int`, and `loss_ref` are computed and combined into `loss_ccc`. |
| Is causal/subspace loss still computed? | Yes, ISR, orthogonality, and invariance losses are computed. |
| Are contrastive/prototype/regularization losses still computed? | Yes, cross-domain contrastive, intervention NT-Xent, reference SupCon, residual-bank responses, ISR, orthogonality, invariance, and optional GT are computed. |
| Do those losses update the shared backbone? | If their effective weights are non-zero, yes: they are attached to `z`/`m` from the shared featurizer. In the analyzed run the PICCL weights were zero, so they did not contribute, but DCCL losses were also omitted from total loss. |
| Are PICCL parameters added to the optimizer? | In normal PICCL, yes: three extra parameter groups for subspace, mediator, and gate are added. In strict bypass, no. |
| Does PICCL forward change BatchNorm stats? | It can execute extra featurizer passes on `all_x_2`; with train-mode BN this can alter running stats. The project defaults `freeze_bn=true`; strict bypass skips PICCL-specific forward paths. |
| Is residual scale only applied at predict? | No. It is applied in training update for residual-gate fusion and also in `predict_embed`. |
| With `scale=0`, is classification feature equal to original DCCL feature? | Yes in residual-gate mode; logs show `feature_delta_norm=0`, `feature_delta_ratio=0`, and `original_fused_cosine=1.0`. |

## Fairness/configuration table

| Field | DCCL | PICCL scale=0 |
| --- | --- | --- |
| Command name | `stage0_dccl_seed0` | `stage0_piccl_scale0_seed0` |
| Dataset | PACS | PACS |
| Algorithm | DCCL | PICCL |
| Test envs | `[0], [1], [2], [3]` sweep | same |
| Seed / trial seed | 0 / 0 | 0 / 0 |
| Hparams seed | default hparams | default hparams plus PICCL overrides |
| Backbone | resnet50 | resnet50 |
| Pretrained | true | true |
| Batch size | 32 | 32 |
| Steps / checkpoint | 500 / 100 | 500 / 100 |
| LR / WD | `5e-5` / `0.0` | same |
| Optimizer | adam | adam |
| Scheduler | none in optimizer config | same |
| Holdout fraction | 0.2 | 0.2 |
| Class balanced | false | false |
| Data augmentation | true, CutMix `aug=0` | same |
| Num workers | trainer default | same |
| Deterministic | true | true |
| AMP | not enabled in logs | not enabled in logs |
| DCCL weights | `l=1`, `l_layer=1`, `l_d=0.01` effective | logged but not added by PICCL replacement loss when PICCL weights are zero |
| PICCL weights | n/a | `piccl_ccc_weight=0`, `piccl_isr=0`, `piccl_orth=0`, `piccl_inv=0`, `connectivity=1`, `int=0.1`, `ref=1` computed |
| Residual scale | n/a | `0` confirmed in JSONL hparams |
| Fusion mode | n/a | `residual_gate` |
| Delayed/warmup | n/a | delayed=0, loss warmup=0, feature warmup=0, alpha warmup/ramp active |
| Optimizer groups | 6 | 9 before this fix; strict bypass now 6 |
| Params | DCCL modules only | DCCL + PICCL modules before this fix; strict bypass excludes PICCL params |

The configurations are not fully equivalent because `algorithm`, optimizer
parameter groups, constructed modules, and effective training objective differ.

## Quantitative checkpoint comparison

The first difference appears at checkpoint step 0: `test_out` is 0.178484 for
DCCL and 0.180929 for PICCL scale=0, while both have identical `ce_loss`
1.940914 and identical `pre_cl_loss` 5.112845.  PICCL's total loss is only CE
(1.940914), whereas DCCL's total loss is 12.331758.

| Step | DCCL test_out | PICCL test_out | Delta | DCCL loss | PICCL loss | DCCL CE | PICCL CE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.178484 | 0.180929 | +0.002445 | 12.331758 | 1.940914 | 1.940914 | 1.940914 |
| 100 | 0.858191 | 0.873472 | +0.015281 | 8.762183 | 0.251699 | 0.314906 | 0.251699 |
| 200 | 0.821516 | 0.828851 | +0.007335 | 7.982994 | 0.131013 | 0.098144 | 0.131013 |
| 300 | 0.664968 | 0.689172 | +0.024204 | 7.685422 | 0.087150 | 0.055990 | 0.087150 |
| 400 | 0.773248 | 0.691720 | -0.081529 | 7.531473 | 0.061146 | 0.028525 | 0.061146 |
| 500 | 0.788535 | 0.707006 | -0.081529 | 7.462093 | 0.052529 | 0.021585 | 0.052529 |

PICCL logged `weighted_loss_ccc=weighted_loss_isr=weighted_loss_orth=weighted_loss_inv=0` at every checkpoint in this scale-0 run, but still logged non-zero unweighted `loss_cross`, `loss_int`, and `loss_ref`.

## First-difference localization

The fixed-batch script now checks four modes:

- A: DCCL
- B: PICCL with `use_piccl=false`
- C: PICCL with `use_piccl=true`, `residual_scale=0`, auxiliary path active
- D: PICCL strict bypass

The diagnostic expectation is that C first differs from A at `piccl_loss` and
therefore at total loss/backbone gradients, while D has no first difference.

## Code changes and new semantics

`residual_scale` remains a feature-fusion strength only.  It is intentionally
not tied to auxiliary loss weights.

A new boolean hparam `piccl_strict_bypass` defines Stage-0 regression mode:

1. PICCL-specific update returns to `DCCL.update`.
2. Prediction embeddings return raw DCCL featurizer output.
3. PICCL parameter groups are excluded from the optimizer.
4. Optimizer group count is the DCCL count (6), not PICCL's normal 9.
5. Normal `use_piccl=true` behavior and `residual_scale=0` auxiliary-regularizer semantics are preserved.

## Next Stage-0 commands

Run only smoke/fixed-batch first; do not start HPO until strict bypass matches
DCCL on fixed batch and 1-step smoke tests.

```bash
python train_all.py stage0_dccl_seed0 --dataset PACS --algorithm DCCL --model resnet50 --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 1 --steps 1 --data_dir /home/hooasia/lgg/data/repro_dccl_data
python train_all.py stage0_piccl_scale0_aux_seed0 --dataset PACS --algorithm PICCL --model resnet50 --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 1 --steps 1 --data_dir /home/hooasia/lgg/data/repro_dccl_data piccl_fusion_mode=residual_gate piccl_residual_scale=0 piccl_ccc_weight=1 piccl_isr_weight=0.1 piccl_orth_weight=0.001 piccl_inv_weight=0.05
python train_all.py stage0_piccl_strict_bypass_seed0 --dataset PACS --algorithm PICCL --model resnet50 --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 1 --steps 1 --data_dir /home/hooasia/lgg/data/repro_dccl_data piccl_strict_bypass=true piccl_fusion_mode=residual_gate piccl_residual_scale=0
```

After the 1-step strict-bypass result matches DCCL, repeat with
`--checkpoint_freq 100 --steps 500`.  HPO should start only after that 500-step
strict-bypass regression passes.
