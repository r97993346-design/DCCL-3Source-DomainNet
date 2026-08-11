# CIPT + DCCL

`CIPTDCCL` is a separate algorithm; the existing `DCCL` implementation and
defaults are not changed. It freezes OpenAI CLIP's image/text encoders and
trains two linear causal-decomposition adapters, one cross-attention TDA layer,
learnable prompt context tokens, and the original DCCL projection/regularizer
modules.

The training path is:

1. CLIP encodes the original and independently augmented images as `v` and
   `v_aug`; the same linear decomposition produces `(e, s)` and
   `(e_aug, s_aug)`.
2. The unchanged `SupConLoss` receives normalized
   `projection_head(e)` / `projection_head(e_aug)`. Thus its augmentation and
   same-label positive mask are unchanged, and no TDA output enters DCCL.
3. The single TDA cross-attention layer makes `K` interventions from `e`.
   Their cosine similarities to learnable CLIP class prompts produce `K` sets
   of logits. Mean-per-intervention CE is the only final task classification
   loss. Inference returns the mean of those `K` logits.
4. The implemented objective is
   `L_c + cipt_beta*L_de + cipt_gamma*L_ind +`
   `cipt_contrastive_weight*L_DCCL + l_layer*pre_cl + l_d*reg_loss`.

An explicit existing checkpoint makes loading offline-safe (the loader never
falls back to a model-name download when `--cipt_clip_path` is supplied):

```bash
cd DCCL/DCCL
python train_all.py cipt_smoke --dataset PACS --algorithm CIPTDCCL \
  --data_dir /path/to/data --test_envs 0 --steps 5 --checkpoint_freq 5 \
  --cipt_clip_backbone ViT-B/16 --cipt_clip_path /path/to/ViT-B-16.pt \
  --cipt_debug_shapes
```

PACS experiment (art-painting as the held-out domain):

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
