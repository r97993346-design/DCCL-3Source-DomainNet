# Frozen 80-template class prototypes for E/S decomposition

Branch: `feature/cipt-fixed80-class-prototype-ablation`

Only **classification class text** changes; the environment/TDA bank remains
independent. The existing `learnable` path is the default and unchanged.
`ensemble80` uses the already normalized `b5b_text_bank` features, takes
their mean for each class, and L2-normalizes the resulting 80-template class
prototype. Prompts and all CLIP encoders are frozen; E/S adapters still train.

## E/S-only experiment

From `DCCL/DCCL`:

```bash
CUDA_VISIBLE_DEVICES=0 python train_all.py pacs_es_fixed80_swad \
  config_cipt_sigm_mcc.yaml config_cipt_fixed80_es.yaml \
  --dataset PACS --algorithm CIPTDCCL \
  --data_dir /root/autodl-tmp/data/repro_dccl_data \
  --output_root /root/autodl-tmp/train_output/ablation/es_fixed80 \
  --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 --aug 0 \
  --cipt_clip_path /root/autodl-tmp/models/ViT-B-16.pt \
  --lr 5e-5 --batch_size 32 \
  --cipt_template_mode b5c \
  --cipt_env_inference e_only \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --swad LossValley
```

## IND=5 variant

Use the same command with a **different** run name/output directory and append:

```bash
  --cipt_use_ind true --cipt_gamma 5
```

## Comparison / testing

The original E/S with learned prompt uses `config_cipt_sigm_mcc.yaml`
without the fixed80 override and with `--cipt_use_de true
--cipt_e_cls_weight 1 --cipt_s_uniform_weight 1 --cipt_use_ind false
--cipt_use_tda false --cipt_env_kl_weight 0 --cipt_use_contrastive false`.

Inspect the startup log for:
`CIPTDCCL class_text_mode=ensemble80 (class prototypes only)`.

CPU-only unit tests:

```bash
python -m unittest discover -s tests -p 'test_cipt_fixed80_class_features.py'
```

This is a *source-trained DG E/S model using fixed class prototypes*, NOT
a zero-shot evaluation. The `cipt_template_mode` flag controls TDA's
intervention bank and is not the fixed class-prototype switch.
