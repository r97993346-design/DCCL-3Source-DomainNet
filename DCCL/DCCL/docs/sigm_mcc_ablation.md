# SIGM + MCC: class-agnostic environment consistency

Base: `feature/cipt-class-agnostic-env-kl`.
New branch: `feature/cipt-class-agnostic-sigm-mcc`.
The old `config_cipt_env_kl.yaml` / `scripts/run_cipt_env_kl.sh` are
**unchanged** and still select original TDA. New experiments use
`config_cipt_sigm_mcc.yaml` / `scripts/run_cipt_sigm_mcc.sh`.

## Precisely what changed

Only the environment-fusion operator is switchable:
- `--cipt_fusion_mode tda`: existing 1-token cross-attention + residual LayerNorm.
- `--cipt_fusion_mode sigm`: Text-IF-inspired scale/shift modulation:
  `[gamma_k,beta_k] = MLP(t_k)`,
  `z_{ik} = (1 + scale*gamma_k) * E_i + scale*beta_k`.
  For adaptive selection, candidate SIGM features replace candidate TDA
  features in exactly the same label-free safe-and-diverse selector.

The CLIP image/text encoders remain frozen; learned class prompts, both E/S
identity-initialized adapters, b5c contexts, selection rules, neighbor-retention
contrastive learning on original E, optimizer type, and the complete loss remain
unchanged. SIGM parameters are added to the optimizer after replacing TDA.

The MLP final layer has small **nonzero** random weights and zero bias
(`cipt_sigm_init_std=0.02`) instead of zero weights. Exact zero initialization
makes all K outputs E, MCC=0, and the entire prompt-only training signal
vanish at initialization. The initialization magnitude and
`cipt_sigm_scale` can be tested independently.

**Important degeneracy**: MCC only minimizes differences among prompt
predictions. Even with nonzero initialization it can learn to ignore prompts;
no feature-diversity or extra label-CE loss was added because the experiment
is intended to preserve the original objective. Inspect the diagnostics.

## Exactly the same objective as the base env_kl branch

```
L = lambda_E * CE(h(E),y)
  + lambda_S * KL(U || h(S))
  + gamma * L_IND(E,S)
  + lambda_con(t) * L_neighbor(E)
  + lambda_kl * L_MCC
L_MCC = 1/[B*K*(K-1)] * sum_i sum_(j != k) KL(p_ij || p_ik)
p_ik = softmax(h(z_ik))
```

Every ordered pair is included; no detach on either KL input, no original-E
prediction in the KL pairs, and no augmented-feature label CE. Default
`lambda_E=1, lambda_S=4, gamma=5, lambda_con(max)=1.5,
lambda_kl=0.1`. Contrastive-only warmup remains at 500 steps.
The KL weight 0.1 is an **unvalidated starting point**, not an optimum.
The E-only base CE weight differs from the historical source no-TDA baseline
which counted E CE more than once: use fixed weights in controlled comparisons.

## PACS / VLCS commands

Execute from repository `DCCL/DCCL`, via `bash` (GitHub-created shell
scripts do not need an executable filesystem permission in that form).

```bash
# SIGM + MCC; fixed seed, lr 5e-5, no image aug, SWAD enabled
CUDA_VISIBLE_DEVICES=0 bash scripts/run_cipt_sigm_mcc.sh PACS

# Exact original TDA control with the same new preset and weights
CIPT_RUN_NAME=pacs_tda_mcc CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_cipt_sigm_mcc.sh PACS --cipt_fusion_mode tda

# Previous TDA + MCC configuration (historical config kept unchanged)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_cipt_env_kl.sh PACS
```

To run VLCS replace `PACS` with `VLCS`.
Set `CIPT_DATA_DIR`, `CIPT_CLIP_PATH`, and `CIPT_RUN_NAME` as needed.

## Controlled ablations

**Fusion operator** (same `lambda_kl`, seed, prompts and settings):
```bash
CIPT_RUN_NAME=pacs_tda_kl01 CUDA_VISIBLE_DEVICES=0 bash scripts/run_cipt_sigm_mcc.sh PACS --cipt_fusion_mode tda
CIPT_RUN_NAME=pacs_sigm_kl01 CUDA_VISIBLE_DEVICES=0 bash scripts/run_cipt_sigm_mcc.sh PACS --cipt_fusion_mode sigm
```

**KL weight sweep** (same SIGM and other weights):
```bash
for w in 0 0.01 0.05 0.1 0.2; do
  CIPT_RUN_NAME="pacs_sigm_kl_${w}" CUDA_VISIBLE_DEVICES=0 \
    bash scripts/run_cipt_sigm_mcc.sh PACS --cipt_env_kl_weight "$w"
done
```
At `lambda_kl=0`, `env_kl_active=False` deliberately bypasses prompt
selection, SIGM/TDA and its training gradient (E classification, S uniformity,
IND and contrastive remain). Therefore this **KL-off** control is NOT a
trained-but-unconstrained SIGM. Such a run is meaningful for testing the
impact of adding MCC, not for evaluating a trained SIGM under fused inference.

**Text selection**:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_cipt_sigm_mcc.sh PACS --cipt_selector_mode random
CUDA_VISIBLE_DEVICES=0 bash scripts/run_cipt_sigm_mcc.sh PACS --cipt_selector_mode adaptive
```

**Inference** (three options):
- `--cipt_env_inference e_only` (default): `h(E)`; no text branch at test time.
- `--cipt_env_inference mean_logits`: mean prompt classification logits.
- `--cipt_env_inference mean_probs`: log of mean prompt softmax probability.

```bash
CIPT_RUN_NAME=pacs_sigm_eonly CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_cipt_sigm_mcc.sh PACS --cipt_env_inference e_only
CIPT_RUN_NAME=pacs_sigm_logits CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_cipt_sigm_mcc.sh PACS --cipt_env_inference mean_logits
CIPT_RUN_NAME=pacs_sigm_probs CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_cipt_sigm_mcc.sh PACS --cipt_env_inference mean_probs
```

**Fairness warning**: These three commands run three independent trainings.
Because SWAD selects checkpoints using validation predictions, changing the
inference mode during training also changes model selection. For a **strict
same-checkpoint inference ablation**, load ONE trained `env_kl` checkpoint,
switch `model.inference_mode` between `e_only`, `mean_logits`,
`mean_probs`, and run the same evaluation loader. The implementation
supports switching at runtime, but no standalone checkpoint evaluation CLI
has been added. Do not report separately trained SWAD-mode runs as a
same-checkpoint inference comparison.

## Diagnostics logged each step

- `env_kl_loss`, `env_kl_term`, `env_kl_w`: unweighted / weighted MCC.
- `env_shift`: mean cosine distance between normalized E and z.
- `env_spread`: average squared deviation among normalized prompts.
- `env_l2_shift`: mean unnormalized ||z-E||_2.
- `env_spread_l2`: average squared deviation among raw z.
- `env_pred_agreement`: fraction of K prompt argmax predictions matching E.
- `sigm_active`, `inference_fused`, `env_active`: mode indicators.
- `e_cls_loss`, `s_uni_loss`, `cipt_ind_loss`, `dccl_contrastive_loss`:
  original training components.

Monitor `env_spread` and `env_l2_shift` alongside performance and MCC.
Very small `env_spread` may indicate prompt collapse; high `env_shift` with
bad agreement may indicate semantic drift. Diagnostics by themselves do not
prove an intervention represents a physically meaningful environment.

## Verification

```bash
python -m unittest discover -s tests -p 'test_cipt_env_kl.py' -v
python -m unittest discover -s tests -p 'test_cipt_sigm_mcc.py' -v
```

These are lightweight CPU tests using toy CLIP substitutes. They validate
formula/gradients/optimizer inclusion, switches, and inference aggregation.
Actual PACS/VLCS training and speed/memory have NOT been verified here.
