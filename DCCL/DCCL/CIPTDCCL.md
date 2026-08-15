# CIPT + DCCL

`CIPTDCCL` is a separate algorithm; the existing `DCCL` implementation and
defaults are not changed. It freezes OpenAI CLIP's image/text encoders and
trains two linear causal-decomposition adapters, one cross-attention TDA layer,
learnable prompt context tokens, and (unless pure CIPT is enabled) the original
DCCL projection/regularizer modules.

## Default CIPT + DCCL path

The existing fusion path is preserved when `cipt_pure: false`:

1. CLIP encodes the original and independently augmented images as `v` and
   `v_aug`; the same linear decomposition produces `(e, s)` and
   `(e_aug, s_aug)`.
2. The unchanged `SupConLoss` receives normalized
   `projection_head(e)` / `projection_head(e_aug)`. Thus its augmentation and
   same-label positive mask are unchanged, and no TDA output enters DCCL.
3. The TDA cross-attention layer makes `K` interventions from `e`.
   Their cosine similarities to learnable CLIP class prompts produce `K` sets
   of logits. Mean-per-intervention CE is the final task classification loss.
4. The fused objective is
   `L_cls + cipt_beta*L_de + cipt_gamma*L_ind +`
   `cipt_contrastive_weight*L_DCCL + l_layer*pre_cl + l_d*reg_loss`.

## Pure CIPT reproduction path

Set `--cipt_pure true` to run a paper-aligned CIPT baseline. This path is
intentionally isolated from DCCL and keeps the fusion path above unchanged.

Pure mode makes the following effective changes:

- objective: `L_cls + cipt_beta*L_de + cipt_gamma*L_ind` only;
- `L_ind` is computed only on `(e, s)` from the original image;
- `x_2` / augmented views are not consumed by the algorithm update;
- DCCL SupCon, pretrained contrastive anchoring, and representation regularizer
  are disabled and their modules are frozen;
- causal and spurious linear adapters are identity-initialized;
- the intervention bank is forced to B5b, the class-conditioned ImageNet prompt
  bank used by the released CIPT implementation;
- TDA uses 8 attention heads;
- inference scores each candidate class with its own B5b intervention contexts
  and averages over `K` intervention logits.

For domain generalization the intended CIPT settings are `beta=4`, `gamma=5`,
and `K=4`.

Example pure CIPT run on PACS:

```bash
cd DCCL/DCCL
CUDA_VISIBLE_DEVICES=0 python train_all.py cipt_pure_pacs_a \
  --dataset PACS --algorithm CIPTDCCL --data_dir /path/to/data \
  --test_envs 0 --deterministic --trial_seed 0 --seed 0 \
  --cipt_clip_backbone ViT-B/16 --cipt_clip_path /path/to/ViT-B-16.pt \
  --cipt_beta 4 --cipt_gamma 5 --cipt_k 4 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_pure true
```

The pure path overrides the effective intervention mode to B5b, TDA heads to 8,
and all DCCL-only loss weights to zero, so callers do not need to pass
`--cipt_contrastive_weight 0 --l_layer 0 --l_d 0` separately.

## Offline-safe CLIP loading

An explicit existing checkpoint makes loading offline-safe (the loader never
falls back to a model-name download when `--cipt_clip_path` is supplied):

```bash
cd DCCL/DCCL
python train_all.py cipt_smoke --dataset PACS --algorithm CIPTDCCL \
  --data_dir /path/to/data --test_envs 0 --steps 5 --checkpoint_freq 5 \
  --cipt_clip_backbone ViT-B/16 --cipt_clip_path /path/to/ViT-B-16.pt \
  --cipt_debug_shapes
```

## Existing fused PACS example

```bash
cd DCCL/DCCL
CUDA_VISIBLE_DEVICES=0 python train_all.py cipt_dccl_pacs_a \
  --dataset PACS --algorithm CIPTDCCL --data_dir /path/to/data \
  --test_envs 0 --deterministic --trial_seed 0 --seed 0 \
  --cipt_clip_backbone ViT-B/16 --cipt_clip_path /path/to/ViT-B-16.pt \
  --cipt_beta 4 --cipt_gamma 5 --cipt_k 4 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_tda_heads 1 --cipt_contrastive_weight 1
```
