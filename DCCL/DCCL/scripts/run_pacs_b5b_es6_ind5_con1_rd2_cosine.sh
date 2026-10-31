#!/usr/bin/env bash
# PACS four-fold DomainBed run for E/S=6, IND=5, contrastive=1, RD=2 + cosine LR.
# Run from any directory after activating the existing training environment.
# Override GPU_ID, CIPT_DATA_DIR, CIPT_CLIP_PATH, CIPT_OUTPUT_ROOT, or CIPT_PYTHON as needed.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${script_dir}/.."

gpu_id="${GPU_ID:-1}"
data_dir="${CIPT_DATA_DIR:-/home/hooasia/lgg/data/repro_dccl_data}"
clip_path="${CIPT_CLIP_PATH:-/home/hooasia/.cache/clip/ViT-B-16.pt}"
output_root="${CIPT_OUTPUT_ROOT:-train_output/ablation/b5b_es6_ind5_con1_rd2_cosine}"
python_cmd="${CIPT_PYTHON:-python}"

if [[ ! -d "$data_dir" ]]; then
    printf 'Data directory not found: %s\nSet CIPT_DATA_DIR to the existing dataset directory.\n' "$data_dir" >&2
    exit 1
fi
if [[ ! -f "$clip_path" ]]; then
    printf 'CLIP checkpoint not found: %s\nSet CIPT_CLIP_PATH to the existing checkpoint.\n' "$clip_path" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="$gpu_id" "$python_cmd" train_all.py pacs_b5b_es6_ind5_con1_rd2_cosine \
    config_cipt_b5b_es6_ind5_con1_rd2_cosine.yaml \
    --dataset PACS --algorithm CIPTDCCL \
    --data_dir "$data_dir" --output_root "$output_root" \
    --cipt_clip_path "$clip_path" \
    --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 \
    --aug 0 --lr 5e-5 --batch_size 32 \
    --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
    --swad LossValley
