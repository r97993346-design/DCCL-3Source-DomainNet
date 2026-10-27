# All-80 official B5b alignment ablations on feature/cipt-b5b-entropy-mcc-rd

Two additional *direct-E* objectives are implemented on the existing branch;
the earlier entropy40 relative-distance (RD) and MCC modes are unchanged.

All-80 mode: `cipt_env_selection=all`, `cipt_env_topk=80`. It uses the
ordered indices `0,...,79` for every image; it never constructs or calls the
entropy selector or the random selector. The class-conditioned bank consists of
80 official CLIP B5b templates **per class**. The text encoder and bank are frozen.
The true class labels are used to retrieve templates *after* deciding to use
all 80; selection is not label based because there is no selection.

For image `i`, let `e_i = normalize(E_i)` and
`t_ij = normalize(T[y_i,j])`; define the **arithmetic mean**
`m_i = (1/80) sum_j t_ij` (do not renormalize the mean).

- **Second formula** `cipt_env_constraint=center_mse`:
  `L2 = mean_i ||e_i-m_i||^2`
- **Fourth formula** `cipt_env_constraint=mean_template_mse`:
  `L4 = mean_i (1/80)sum_j ||e_i-t_ij||^2`

Both objectives are squared Euclidean distances between normalized individual
features. The mean is deliberately not renormalized in the second formula
because `L4 = L2 + mean_i(1/80)sum_j ||t_ij-m_i||^2`. Frozen text
makes the added variance constant with respect to E, so their E gradients are
identical. The numeric losses are **not** identical; compare accuracy and
gradient numerically instead of comparing reported loss magnitude.

The original third-formula RD is still `cipt_env_constraint=relative_distance`,
which matches cosine similarities of E and the normalized class center to
each selected template. With `cipt_env_selection=all` and topk 80, the
original RD can be evaluated without entropy selection for a three-way study.

## Ready-to-use presets

- `config_cipt_b5b_all80_center_mse.yaml` (second formula)
- `config_cipt_b5b_all80_mean_template_mse.yaml` (fourth formula)

These presets isolate the alignment losses with E CE=1, S uniform=1, IND off,
contrastive off, no TDA/fusion, e_only inference, SWAD LossValley, lr=5e-5.
The existing entropy40 RD preset is not modified.

## Launch on a server in the DCCL working directory

```bash
CUDA_VISIBLE_DEVICES=0 python train_all.py pacs_all80_center_mse_swad \
  config_cipt_b5b_all80_center_mse.yaml \
  --dataset PACS --algorithm CIPTDCCL \
  --data_dir /root/autodl-tmp/data/repro_dccl_data \
  --output_root /root/autodl-tmp/train_output/ablation/rd_all80_forms \
  --cipt_clip_path /root/autodl-tmp/models/ViT-B-16.pt \
  --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 \
  --aug 0 --lr 5e-5 --batch_size 32 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a"

CUDA_VISIBLE_DEVICES=0 python train_all.py pacs_all80_mean_template_mse_swad \
  config_cipt_b5b_all80_mean_template_mse.yaml \
  --dataset PACS --algorithm CIPTDCCL \
  --data_dir /root/autodl-tmp/data/repro_dccl_data \
  --output_root /root/autodl-tmp/train_output/ablation/rd_all80_forms \
  --cipt_clip_path /root/autodl-tmp/models/ViT-B-16.pt \
  --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 \
  --aug 0 --lr 5e-5 --batch_size 32 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a"
```

If using a different server, replace the data/output/CLIP checkpoint paths.
For the original third formula using all 80 templates, launch the existing
`config_cipt_b5b_entropy_rd.yaml` but explicitly override
`--cipt_env_selection all --cipt_env_topk 80`, and disable IND/contrastive to
match the ablation presets. The third formula still uses normalized centers.

## Validation

```bash
python -m unittest discover -s tests -p 'test_cipt_rd_all80_alignment.py'
```

The tests verify direct formula values, variance identity, equal E gradients,
frozen text, no entropy/random selector call, a real small-model update, and
rejection of partial-bank configurations.

New training logs: `env_center_mse_loss`,
`env_mean_template_mse_loss`, `env_all_templates_active`,
`env_topk`. No model structure, inference code, or existing loss is replaced.
