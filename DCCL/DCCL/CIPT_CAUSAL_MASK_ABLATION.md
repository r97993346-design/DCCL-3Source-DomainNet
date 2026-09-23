# CIPT causal-mask / XCorr ablation

This branch adds a switchable causal decomposition path without changing the
default behavior of the parent branch.

## New switches

- `--cipt_decomposition_mode dual_linear|causal_mask`
  - `dual_linear`: original identity-initialized E/S linear adapters.
  - `causal_mask`: sample-specific complementary mask,
    `E=M(V)*V`, `S=(1-M(V))*V`.
- `--cipt_independence_mode cosine|xcorr`
  - `cosine`: original CIPT sample-wise cosine orthogonality.
  - `xcorr`: batch cross-dimensional decorrelation, driving C_ES toward zero.
- `--cipt_mask_hidden_dim 128`: bottleneck width of the mask generator.
- `--cipt_mask_temperature 1.0`: Binary-Concrete/Gumbel-Sigmoid temperature.
- `--cipt_mask_hard` / `--no-cipt_mask_hard`: straight-through binary mask.
  Soft masks are the default and are recommended for initial experiments.
- `--cipt_mask_sparsity_weight 0.0`: weight on mean mask activation.
  A positive value discourages the trivial `M -> 1, S -> 0` solution.

The existing `--cipt_use_de`, `--cipt_use_ind`, `--cipt_use_tda`, and
`--cipt_use_contrastive` switches remain available.

## Loss organization

For causal-mask runs the existing CIPT decomposition semantics are retained:

`L_de = CE(p(Y|E), y) + KL(U || p(Y|S))`.

The independence term is switchable:

- cosine: `L_ind = 0.5 * mean(cos(E,S)^2)`
- xcorr: `L_ind = mean(C_ES^2)`,
  where `C_ES = standardize(E)^T standardize(S) / B`.

The complete base objective becomes:

`L = L_TDA + beta*L_de + gamma*L_ind + lambda_m*mean(M) + L_contrastive`.

The neighborhood-retention contrastive objective is unchanged and remains
applied only to E.

## Recommended comparison sequence

1. Exact parent baseline:
   `dual_linear + cosine + mask_sparsity_weight=0`.
2. Structure-only diagnostic:
   `causal_mask + cosine + mask_sparsity_weight=0`.
3. Causal-mask decomposition:
   `causal_mask + cosine + mask_sparsity_weight=1e-3`.
4. Full proposed decomposition:
   `causal_mask + xcorr + mask_sparsity_weight=1e-3`.

For the first causal-mask run, keep `mask_hard=false` and temperature 1.0.
Tune beta/gamma/sparsity only after verifying that mask diagnostics do not
collapse.

## Logged diagnostics

Every update reports:

- `cipt_mask_loss`
- `cipt_mask_weighted_loss`
- `cipt_mask_mean`, `cipt_mask_std`
- `cipt_mask_min`, `cipt_mask_max`
- existing `mean_e_norm`, `mean_s_norm`, `mean_es_cosine`

A healthy run should not immediately drive `cipt_mask_mean` to either 0 or 1.
