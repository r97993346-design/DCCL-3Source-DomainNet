# CIPT + DCCL

`CIPTDCCL` keeps the existing feature/multiprompt implementation and provides a
minimal switch for running the CIPT reproduction setting without changing the
rest of the training logic.

## Default CIPT + DCCL path

With `cipt_pure: false`, the existing fusion path is unchanged:

1. CLIP encodes the original and independently augmented images as `v` and
   `v_aug`; the same linear decomposition produces `(e, s)` and
   `(e_aug, s_aug)`.
2. `SupConLoss` receives normalized `projection_head(e)` and
   `projection_head(e_aug)`.
3. TDA produces `K` interventions from `e`, which are classified against the
   learnable CLIP class prompts.
4. The objective remains
   `L_cls + cipt_beta*L_de + cipt_gamma*L_ind +`
   `cipt_contrastive_weight*L_DCCL + l_layer*pre_cl + l_d*reg_loss`.

## Minimal CIPT reproduction switch

Set `--cipt_pure true` to change only the two components requested for the CIPT
reproduction experiment:

- disable the DCCL supervised contrastive term by setting the effective
  `cipt_contrastive_weight` to `0`;
- disable the pretrained contrastive anchoring term by setting the effective
  `l_layer` to `0`;
- disable random training augmentation by using the existing `DBT.basic`
  preprocessing instead of `DBT.aug` for the training split.

No other logic is changed by this switch. In particular it does **not** change:

- the selected B5a/B5b/B5c prompt bank;
- `cipt_tda_heads`;
- causal/spurious adapter initialization;
- optimizer construction;
- `l_d` / representation regularization;
- the existing `L_ind` computation;
- SWAD, its early-stop behavior, IID/oracle/last selection, or evaluation logic;
- inference behavior.

The dataset interface is also left unchanged: `x` and `x_2` are still returned.
Because both use deterministic `DBT.basic` when `cipt_pure=true`, they represent
the same non-randomly-augmented view while the existing algorithm code can stay
unchanged.

Example:

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

## Offline-safe CLIP loading

An explicit local checkpoint keeps loading offline-safe:

```bash
cd DCCL/DCCL
python train_all.py cipt_smoke --dataset PACS --algorithm CIPTDCCL \
  --data_dir /path/to/data --test_envs 0 --steps 5 --checkpoint_freq 5 \
  --cipt_clip_backbone ViT-B/16 --cipt_clip_path /path/to/ViT-B-16.pt \
  --cipt_debug_shapes
```
