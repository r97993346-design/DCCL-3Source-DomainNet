# CIPT + DCCL

`CIPTDCCL` is a separate algorithm; the existing `DCCL` implementation and
defaults are not changed. The active implementation is routed through
`domainbed/algorithms/cipt_dccl_official.py` and aligns the CIPT portion with
the public upstream CIPT implementation while retaining the DCCL causal-space
contrastive extension.

## Official-aligned CIPT components

- OpenAI CLIP image/text encoders remain frozen.
- CLIP image embeddings are L2-normalized before causal decomposition.
- DomainBed ImageNet-normalized tensors are re-normalized to OpenAI CLIP input
  statistics inside the CIPT model, so the DCCL data pipeline remains unchanged.
- Causal and spurious adapters are one linear layer each and use identity-weight,
  zero-bias initialization.
- Learnable class prompts use the CoOp-style context initialized from
  `a photo of a`, with 16 context tokens by default.
- TDA uses the OpenAI ImageNet prompt-template bank, formatted with each class
  name. During training, K templates are sampled randomly; inference uses a
  deterministic K-template subset and performs class-conditioned intervention.
- TDA uses one multi-head-attention layer with 8 heads by default. An explicit
  `--cipt_tda_heads 1/2/4/8` value is preserved for controlled ablations.
- `L_de` is causal CE plus `KL(uniform || p_spurious)`.
- `L_ind = 0.5 * mean(cos(e, s)^2)`.
- `L_c` is mean cross-entropy over K intervention-specific predictions.
- CIPT prompt/adapters/TDA use Adam with the upstream default learning rate
  `2.5e-3` and zero weight decay.
- The CIPT optimizer group follows cosine LR decay over the configured DomainBed
  training-step horizon. The DCCL extension parameter group keeps the original
  DCCL learning rate rather than being decayed by the CIPT scheduler.

## SWAD integration

`CIPTDCCL` exposes a lightweight inference-only forward model to SWAD. SWAD
averages only prediction-relevant trainable parameters (causal adapters, prompt
context and TDA). Frozen CLIP/text-encoder parameters and optimizer state are
not traversed by the per-step SWAD averaging loop. This avoids deep-copying or
averaging the full frozen CLIP training object at every update.

## DCCL integration

The DCCL extension is intentionally kept separate from the official CIPT core:

1. CLIP encodes original and independently augmented images as `v` and
   `v_aug`; causal decomposition yields `e` and `e_aug`.
2. The existing `SupConLoss` receives normalized `projection_head(e)` and
   `projection_head(e_aug)`. No TDA output enters this DCCL contrastive branch.
3. The existing feature/multiprompt pretrained anchoring and vector
   regularization terms remain available through `l_layer` and `l_d`.

The objective is:

`L_c + cipt_beta*L_de + cipt_gamma*L_ind +`
`cipt_contrastive_weight*L_DCCL + l_layer*pre_cl + l_d*reg_loss`.

For domain generalization, the upstream CIPT settings are `beta=4`, `gamma=5`,
and `K=4`.

## Smoke test

```bash
cd DCCL/DCCL
CUDA_VISIBLE_DEVICES=0 python train_all.py cipt_smoke \
  --dataset PACS --algorithm CIPTDCCL --data_dir /path/to/data \
  --test_envs 0 --steps 5 --checkpoint_freq 5 \
  --cipt_clip_backbone ViT-B/16 --cipt_clip_path /path/to/ViT-B-16.pt \
  --cipt_beta 4 --cipt_gamma 5 --cipt_k 4 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_tda_heads 8 --cipt_contrastive_weight 1 \
  --cipt_debug_shapes
```

## PACS example

```bash
cd DCCL/DCCL
CUDA_VISIBLE_DEVICES=0 python train_all.py cipt_dccl_pacs_a \
  --dataset PACS --algorithm CIPTDCCL --data_dir /path/to/data \
  --test_envs 0 --deterministic --trial_seed 0 --seed 0 \
  --checkpoint_freq 200 \
  --cipt_clip_backbone ViT-B/16 --cipt_clip_path /path/to/ViT-B-16.pt \
  --cipt_beta 4 --cipt_gamma 5 --cipt_k 4 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_tda_heads 8 --cipt_contrastive_weight 1
```
