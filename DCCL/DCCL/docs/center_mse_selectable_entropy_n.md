# Selectable official-template center alignment

The `center_mse` environment constraint can now use a per-image, frozen-CLIP
entropy-selected top-N teacher center, or all 80 official B5b templates with
no selection. **It never substitutes the global 80-vector center for an
entropy-selected center.** The frozen text bank remains class-conditioned.

`cipt_env_selection=entropy`: choose indices J_i, |J_i|=N (2 <= N <= 80),
by minimum class-prediction entropy using original frozen CLIP image features
V and class-conditional B5b text features; the true label is not used for
selection. Then fetch text `T[y_i, J_i]` with training label y_i and compute
`center_i=mean_{j in J_i} normalize(T[y_i,j])`. Train with
`mean_i ||normalize(E_i)-center_i||^2`. The **selected center is recomputed
per image and per update**. Its mean is not normalized again, matching the
earlier all-80 center MSE definition.

`cipt_env_selection=all` with `cipt_env_topk=80`: bypass both entropy and
random selectors; center is the arithmetic mean of all 80 official templates.

`cipt_env_selection=random` with N between 2 and 80 is additionally
supported as an optional random-top-N comparison; the teacher center uses
the actual sampled indices, not all 80.

`mean_template_mse` remains the original all-80 control. The original
`relative_distance` RD/MCC selection and the existing E-only inference
remain unchanged.

Ready-to-use experiment config: `config_cipt_b5b_center_mse.yaml` (default
entropy Top-40, E=1, S=1, center weight=1, IND/contrastive off, lr=5e-5,
SWAD LossValley).

On the PACS training server, from DCCL/:

```bash
# Entropy-selected N=40 (change 40 to 20 or 60 as needed):
CUDA_VISIBLE_DEVICES=1 python train_all.py pacs_entropy40_selected_center_e1_s1 \
  config_cipt_b5b_center_mse.yaml --dataset PACS --algorithm CIPTDCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --output_root train_output/ablation/center_selectable_gpu1 \
  --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 \
  --aug 0 --lr 5e-5 --batch_size 32 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_env_selection entropy --cipt_env_topk 40

# No entropy: all 80, matching the older center_mse baseline:
CUDA_VISIBLE_DEVICES=1 python train_all.py pacs_all80_center_e1_s1 \
  config_cipt_b5b_center_mse.yaml --dataset PACS --algorithm CIPTDCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --output_root train_output/ablation/center_selectable_gpu1 \
  --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 \
  --aug 0 --lr 5e-5 --batch_size 32 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_env_selection all --cipt_env_topk 80
```

Tests:
`python -m unittest discover -s tests -p 'test_cipt_center_selection.py'`.
