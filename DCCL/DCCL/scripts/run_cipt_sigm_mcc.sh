#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-PACS}"
if (( $# > 0 )); then shift; fi
case "$dataset" in
  PACS|VLCS) ;;
  *) echo "Usage: $0 [PACS|VLCS] [additional train_all.py options]" >&2; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
run_name="${CIPT_RUN_NAME:-${dataset,,}_b5c_sigm_mcc_swad}"

exec python train_all.py "$run_name" config_cipt_sigm_mcc.yaml \
  --dataset "$dataset" \
  --algorithm CIPTDCCL \
  --data_dir "${CIPT_DATA_DIR:-/home/hooasia/lgg/data/repro_dccl_data}" \
  --cipt_clip_path "${CIPT_CLIP_PATH:-/home/hooasia/.cache/clip/ViT-B-16.pt}" \
  --output_root train_output/ablation/sigm_mcc \
  --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 --aug 0 \
  "$@"
