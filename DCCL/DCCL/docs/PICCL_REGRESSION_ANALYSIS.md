# PICCL Stage-0 regression analysis

## Current implementation semantics

PICCL is no longer a replacement objective for DCCL. In normal `use_piccl=true`
mode, the update path is:

```text
raw backbone feature z
  -> causal mediator
  -> residual gate fusion(z, mediator(z))
  -> original DCCL losses
```

The original DCCL terms remain active after fusion and are composed by the shared
`DCCL.update` loss flow: CE, optional two-view CE, SupCon over `x/x_2`,
pre-trained SupCon against `pre_featurizer`, and the `l_d` pre-training
anchoring regularizer. PICCL only overrides the post-backbone feature hook and
the extra causal-loss hook. It adds two causal auxiliary losses: `loss_isr` and
`loss_orth`, weighted independently by `piccl_isr_weight` and
`piccl_orth_weight`.

## Residual-scale semantics from code

| Question | Answer |
| --- | --- |
| Does `residual_scale=0` close feature fusion? | Yes. The gate computes `original + scale * alpha * gate * (mediator-original)`, so `scale=0` gives the original backbone feature for DCCL losses. |
| Does the PICCL module still execute forward/update? | Yes unless `use_piccl=false` or `piccl_strict_bypass=true`. |
| Are connectivity replacement losses computed? | No. `loss_ccc`, `loss_int`, `loss_ref`, and `loss_inv` are no longer part of the active PICCL objective. |
| Is causal/subspace loss still computed? | Yes. Only ISR coverage and orthogonality are added as causal auxiliary losses. |
| Are original DCCL losses still computed? | Yes. Fused features are sent through the DCCL CE, SupCon, pre-trained SupCon, and optional `l_d` anchoring terms. |
| Do causal losses share a scheduler? | No. `piccl_isr_weight` and `piccl_orth_weight` are applied directly and independently. |
| Are PICCL parameters added to the optimizer? | In normal PICCL, yes: subspace, mediator, and residual gate parameter groups are added. In strict bypass, no. |
| Is residual scale only applied at predict? | No. It is applied in training update and in `predict_embed`/forward-model inference. |

## Stage-0 comparison modes

- A: DCCL
- B: PICCL with `use_piccl=false`
- C: PICCL with `use_piccl=true`, `piccl_residual_scale=0`, causal losses active
- D: PICCL with `piccl_strict_bypass=true`

Mode C should have the same fused feature as DCCL when `piccl_residual_scale=0`,
but it can still differ in total loss and gradients if `piccl_isr_weight` or
`piccl_orth_weight` are non-zero. Mode D is the strict bypass mode and should
return to `DCCL.update`, raw DCCL prediction embeddings, and DCCL-only optimizer
groups.

## Recommended smoke commands

```bash
python train_all.py stage0_dccl_seed0 --dataset PACS --algorithm DCCL --model resnet50 --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 1 --steps 1 --data_dir /home/hooasia/lgg/data/repro_dccl_data
python train_all.py stage0_piccl_scale0_aux_seed0 --dataset PACS --algorithm PICCL --model resnet50 --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 1 --steps 1 --data_dir /home/hooasia/lgg/data/repro_dccl_data piccl_residual_scale=0 piccl_isr_weight=0.1 piccl_orth_weight=0.001
python train_all.py stage0_piccl_strict_bypass_seed0 --dataset PACS --algorithm PICCL --model resnet50 --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 1 --steps 1 --data_dir /home/hooasia/lgg/data/repro_dccl_data piccl_strict_bypass=true piccl_residual_scale=0
```
