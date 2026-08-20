# CIPT + DCCL

`CIPTDCCL` has two explicitly separated input/training paths while keeping
the existing prompt bank, TDA, optimizer, SWAD, model selection and inference
logic unchanged.

## Pure CIPT reproduction

Set `--cipt_pure true`.

The training sample contains only one original/basic view `x`; no `x_2` is
created or consumed. The effective DCCL-side weights are:

- `cipt_contrastive_weight = 0`
- `l_layer = 0`
- `l_d = 0`

Therefore the pure objective is:

`L_cls + cipt_beta * L_de + cipt_gamma * L_ind`

No prompt-bank, TDA-head, adapter-initialization, optimizer, SWAD, IID/oracle,
or inference setting is changed by this switch.

## CIPT + DCCL fusion

With `cipt_pure: false`, every training sample has exactly two views of the same
image:

- `x`: `DBT.clip_basic`, the official CLIP Resize + CenterCrop + normalization pipeline;
- `x_2`: `DBT.clip_aug`, using RandomResizedCrop + horizontal flip + ColorJitter + random grayscale + CLIP normalization.

The augmented view has exactly three roles:

1. **SupCon**: `e` and `e_aug` form the two positive views used by the existing
   supervised contrastive loss.
2. **Causal consistency**: the causal features of the original and augmented
   views are aligned with
   `L_cons = mean(1 - cosine(e, e_aug))`. Its default weight is
   `cipt_causal_consistency_weight: 1.0`.
3. **Augmented decomposition loss**: `e_aug` must remain class-discriminative and
   `s_aug` must remain class-uninformative. The code computes `L_de_orig` and
   `L_de_aug`, then uses
   `L_de = 0.5 * (L_de_orig + L_de_aug)` so the existing `cipt_beta` scale is
   preserved.

The augmented view is not used for `L_ind`, TDA classification, pre-CL, or the
representation regularizer. Those remain on the original-image branch.

The fusion objective is:

`L_cls + beta*L_de + gamma*L_ind +`
`cipt_causal_consistency_weight*L_cons +`
`cipt_contrastive_weight*L_DCCL + l_layer*pre_cl + l_d*reg_loss`

## Example pure CIPT run

```bash
cd DCCL/DCCL
CUDA_VISIBLE_DEVICES=0 python train_all.py cipt_repro \
  --dataset PACS --algorithm CIPTDCCL --data_dir /path/to/data \
  --test_envs 0 --deterministic --trial_seed 0 --seed 0 \
  --cipt_clip_backbone ViT-B/16 --cipt_clip_path /path/to/ViT-B-16.pt \
  --cipt_beta 4 --cipt_gamma 5 --cipt_k 4 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_pure true
```
