# Switchable official-style CIPT cosine learning-rate policy

Upstream (pinned official CIPT reproduction at `repro/cipt-official-domainbed`):
Adam, initial LR 0.0025, weight_decay=0.0, CosineAnnealingLR stepped once
after every epoch across 30 epochs, global batch_size 64.

This DomainBed implementation uses a *step-based* trainer, not the upstream
30-epoch loader. Therefore `cipt_lr_schedule=cosine` uses the same PyTorch
cosine annealing shape but advances once **after every** `algorithm.update()`
over the entire actual optimizer-step budget. In `train_all.py`,
`--steps 3000 --checkpoint_freq 100` gives 3001 loop iterations.
The cosine scheduler uses `T_max=3001` and logs the LR **applied** to each
update as `lr` in `results.jsonl`. Each held-out target gets a newly built
optimizer and scheduler starting from the same initial LR.

Options (only CIPTDCCL `env_anchor`):
- `cipt_lr_schedule: constant` (default): old training behavior unchanged.
- `cipt_lr_schedule: cosine`: cosine decay from `lr` to `cipt_lr_min`.
- `cipt_lr_min: 0.0` default for upstream-style cosine tail.

Use `config_cipt_lr_cosine.yaml` as an **additional** config following the
experiment config:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train_all.py \
  pacs_all80_center_e1_s1_lr5e5_cosine_3000 \
  config_cipt_b5b_all80_center_mse.yaml config_cipt_lr_cosine.yaml \
  --algorithm CIPTDCCL --dataset PACS \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --output_root train_output/ablation/cipt_cosine_lr \
  --deterministic --trial_seed 0 --seed 0 \
  --steps 3000 --checkpoint_freq 100 --aug 0 \
  --lr 5e-5 --batch_size 32 \
  --num_workers 2 --eval_num_workers 0 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_class_text_mode learnable \
  --cipt_env_constraint center_mse --cipt_env_selection all \
  --cipt_env_topk 80 --cipt_env_loss_weight 1.0 \
  --cipt_e_cls_weight 1.0 --cipt_s_uniform_weight 1.0 \
  --cipt_use_ind false --cipt_use_contrastive false \
  --swad LossValley
```

For the official numerical LR test, *also* change `--lr 0.0025` and, if
testing the full upstream recipe, `--batch_size 64` and weight decay 0;
however these are large independent changes and no longer isolate the effect
of the scheduler. Also the upstream model has different losses, architecture
usage and no SWAD, so it is NOT an exact end-to-end official reproduction.

This hook changes neither E/S decomposition nor RD nor evaluation/model
selection. No new scheduler is created for `legacy`, `env_kl`, or
`clip_ce` with default `constant`.
