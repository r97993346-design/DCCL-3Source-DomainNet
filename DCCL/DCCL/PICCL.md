# PICCL v6: Causal Features Before Original DCCL

PICCL is an independent algorithm. The original `DCCL` class and its
`SupConLoss` remain unchanged.

## Data flow

1. Original and augmented images share the original DCCL featurizer and produce
   pooled features `z` and `z_int`.
2. PIRE and a class/domain EMA residual bank learn a low-rank
   intervention-sensitive subspace `Q` using `loss_isr` and `loss_orth`.
3. The causal mediator produces
   `m = z - beta * Proj_Q(z)` and
   `m_int = z_int - beta * Proj_Q(z_int)`.
4. `m` and `m_int` replace only the final features entering DCCL's classifier,
   projection head and pretrained-anchor contrast. DCCL's masks, temperatures,
   CE, SupCon, `pre_cl_loss`, `reg_loss`, TN, CutMix and `sample_d` branches keep
   their original definitions.

The residual bank and ISR losses are active from step zero. `beta` remains zero
during warm-up and then ramps linearly, so the sensitive subspace is learned
before it perturbs DCCL features.

## Reliability

Reliable contrast is a separate, opt-in stage. For a cross-domain same-class
pair, reliability is one minus the fraction of the original feature difference
explained by `Q`. Consequently, nuisance-heavy pairs receive less positive
weight and invariant pairs retain full weight. Self-augmentation and
same-domain positives keep weight one; negative logits and masks are unchanged.
The original DCCL `SupConLoss` is not modified.

## Equivalence switches

- Selecting `--algorithm PICCL` enables causal projection by default.
- `PICCL --use_piccl false` directly delegates training and inference to DCCL.
- With `beta=0` and zero auxiliary weights, the causal task path is an identity.
- `piccl_use_reliable_contrast=false` uses the original unweighted DCCL SupCon.

For SWAD, backbone/classifier parameters are averaged normally. The orthonormal
`Q` and `beta` are exported as buffers and copied from the latest model rather
than averaging the raw basis.
