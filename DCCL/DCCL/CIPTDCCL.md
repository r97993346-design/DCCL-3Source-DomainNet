# CIPT + Single-View Direct Causal Contrastive Learning

This branch is the C ablation for separating contrastive-learning gains from
augmentation-view gains. It is based on `feature/cipt-causal-contrastive-no-proj`
but removes the augmented contrastive view entirely.

## Pure CIPT

Set `--cipt_pure true`.

The objective is:

`L_CIPT = L_cls + beta * L_de + gamma * L_ind`

Only the original CLIP-preprocessed image is consumed.

## Single-view direct causal contrastive mode

With the branch defaults:

- `cipt_pure: false`
- `cipt_single_view_contrastive: true`
- `cipt_causal_contrastive_weight: 0.1`
- `cipt_contrastive_warmup_steps: 500`

training consumes only one deterministic original view:

`x = CLIP Resize + CenterCrop + RGB + normalization`

There is no `x_2`, RandomResizedCrop, horizontal flip, ColorJitter, grayscale,
or augmented causal representation in the training path.

The original view follows normal CIPT:

`x -> frozen CLIP -> (e, s) -> L_cls + beta*L_de + gamma*L_ind`

The same causal representation `e` is also used directly for supervised
contrastive learning. There is no projection head.

For anchor `i`, the positive set is:

`P(i) = {j | j != i and y_j == y_i}`

All other non-self samples are contrastive negatives. A sample with no
same-class peer in the current merged minibatch is skipped as an anchor. It is
still retained in the contrast pool and can act as a negative for valid
anchors. If the whole minibatch contains no same-class pair, `L_con = 0` and
the update reduces to pure CIPT for that step.

The objective is:

`L_total = L_CIPT + lambda_eff * L_con_single_view`

with linear warmup:

`lambda_eff = lambda_max * min(1, step / warmup_steps)`

This branch therefore measures class-level direct causal supervised contrastive
learning without using an augmentation-induced positive pair.

## Logged diagnostics

The training step additionally reports:

- `dccl_contrastive_loss`
- `contrastive_weight_eff`
- `contrastive_valid_anchors`
- `contrastive_valid_anchor_ratio`
- `contrastive_positive_links`

These values make it possible to verify how much of each minibatch actually
contributes anchors to the single-view contrastive objective.

## PACS C-ablation example

```bash
cd DCCL/DCCL
CUDA_VISIBLE_DEVICES=0 python train_all.py pacs_b5c_singleview_cl01 \
  --dataset PACS --algorithm CIPTDCCL --data_dir /path/to/data \
  --deterministic --trial_seed 0 --seed 0 \
  --cipt_clip_backbone ViT-B/16 --cipt_clip_path /path/to/ViT-B-16.pt \
  --cipt_beta 4 --cipt_gamma 5 --cipt_k 4 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_template_mode b5c \
  --cipt_causal_contrastive_weight 0.1 \
  --cipt_contrastive_warmup_steps 500
```
