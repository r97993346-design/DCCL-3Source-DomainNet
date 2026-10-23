#!/usr/bin/env bash
# Run from any directory after activating the existing training environment.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${script_dir}/.."

data_dir="${CIPT_DATA_DIR:-/home/hooasia/lgg/data/repro_dccl_data}"
clip_path="${CIPT_CLIP_PATH:-/home/hooasia/.cache/clip/ViT-B-16.pt}"
output_root="${CIPT_OUTPUT_ROOT:-train_output/ablation/b5b_entropy_mcc_rd}"
python_cmd="${CIPT_PYTHON:-python}"

if [[ ! -d "$data_dir" ]]; then
    printf 'Data directory not found: %s\nSet CIPT_DATA_DIR to the existing dataset directory.\n' "$data_dir" >&2
    exit 1
fi
if [[ ! -f "$clip_path" ]]; then
    printf 'CLIP checkpoint not found: %s\nSet CIPT_CLIP_PATH to the existing checkpoint.\n' "$clip_path" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=1 "$python_cmd" train_all.py pacs_rd2_entropy40_selected_center_gpu1 \
    config_cipt_b5b_entropy_rd.yaml \
    --dataset PACS --algorithm CIPTDCCL \
    --data_dir "$data_dir" --output_root "$output_root" \
    --cipt_clip_path "$clip_path" \
    --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 \
    --aug 0 --lr 5e-5 --batch_size 32 \
    --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
    --cipt_objective env_anchor --cipt_env_constraint relative_distance \
    --cipt_template_mode b5b --cipt_selector_mode random \
    --cipt_env_selection entropy --cipt_env_topk 40 \
    --cipt_rd_anchor_mode selected --cipt_env_loss_weight 2.0 \
    --cipt_use_de true --cipt_beta 1.0 \
    --cipt_e_cls_weight 1.0 --cipt_s_uniform_weight 1.0 \
    --cipt_use_ind false --cipt_gamma 0.0 \
    --cipt_use_tda false --cipt_env_inference e_only \
    --cipt_env_kl_weight 0.0 --cipt_fused_cls_weight 0.0 \
    --cipt_pure false --cipt_use_aug_view false \
    --cipt_use_contrastive true --cipt_contrastive_type neighbor_retention \
    --cipt_neighbor_k 10 --cipt_neighbor_alpha 4.0 \
    --cipt_contrastive_temperature 0.07 \
    --cipt_causal_contrastive_weight 1.5 \
    --cipt_contrastive_warmup_steps 500 \
    --swad LossValley
