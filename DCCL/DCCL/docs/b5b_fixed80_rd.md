# Frozen B5b ensemble with all-80 relative-distance supervision

Preset: `config_cipt_b5b_fixed80_rd.yaml`.

The original learnable-prompt and entropy-selected experiments retain their
defaults. The new switches apply only to `cipt_objective=env_anchor`:

| Option | Default | This experiment |
| --- | --- | --- |
| `cipt_class_text_mode` | `learnable` | `b5b_ensemble` |
| `cipt_env_selection` | `entropy` | `all` |
| `cipt_env_topk` | `40` | `80` (required with `all`) |
| `cipt_env_constraint` | `mcc` | `relative_distance` |

Let `T[c,j]` be the existing frozen, individually normalized CLIP encoding of
official B5b template `j` containing class `c`. The full bank is `[C,80,D]`.
Compute once, using all 80 vectors before the final normalization:

`A[c] = normalize(mean_j T[c,j])`.

Training CE and prediction both use:

`logits[i,c] = exp(CLIP.logit_scale) * dot(normalize(E[i]), A[c])`.

S uniformity uses the same fixed class centers, replacing E with S. The
PromptLearner is frozen, excluded from the optimizer and never called by
training classification or prediction. Its implementation is unchanged.

RD retains every template relationship, rather than replacing it with a
feature MSE:

`L_RD = mean_{i,j=0..79} (cos(E[i], T[y[i],j]) - cos(A[y[i]], T[y[i],j]))^2`.

The full-bank path uses indices `0..79` directly. It does not construct or call
the entropy selector, compute entropy, or call the inherited random/adaptive
selectors. `cipt_selector_mode=random` only satisfies the inherited constructor.
Prediction needs no labels and performs no environment-template selection.

The preset optimizes only the E/S adapters, with:

`L = CE_E + KL(U || p_S) + L_RD`.

IND, contrastive learning, TDA/SIGM, old environment KL and fused CE are disabled.
SWAD is unchanged. Logged indicators should include `entropy_selector_active=0`,
`env_all_templates_active=1`, `class_text_ensemble_active=1`, `env_topk=80`,
and both contrastive loss and its effective coefficient equal to zero.

From `DCCL/DCCL`, on the local server:

```bash
CUDA_VISIBLE_DEVICES=0 python train_all.py pacs_b5b_fixed80_rd_nocon_swad \
  config_cipt_b5b_fixed80_rd.yaml \
  --dataset PACS --algorithm CIPTDCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --output_root train_output/ablation/b5b_fixed80_rd \
  --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 \
  --aug 0 --lr 5e-5 --batch_size 32 \
  --cipt_class_text_mode b5b_ensemble \
  --cipt_env_selection all --cipt_env_topk 80 \
  --cipt_e_cls_weight 1.0 --cipt_s_uniform_weight 1.0 \
  --cipt_env_loss_weight 1.0 \
  --cipt_use_ind false --cipt_use_tda false \
  --cipt_use_contrastive false --cipt_causal_contrastive_weight 0.0 \
  --swad LossValley
```

For a controlled comparison using all-80 RD with the original learnable
classifier, override only `--cipt_class_text_mode learnable`.

Validation uses the real algorithm constructors, losses, optimizer and SWA,
with only CLIP weights/tokenization replaced by a small frozen CPU fixture:

```bash
python -m unittest discover -s tests -p 'test_cipt_*.py'
python tests/test_cipt_fixed80_rd.py --smoke
```
