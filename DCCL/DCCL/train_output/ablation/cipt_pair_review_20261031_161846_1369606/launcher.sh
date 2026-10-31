#!/usr/bin/env bash
# Saved DomainBed original CIPT vs saved upstream original CIPT, lr=5e-5.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${CIPT_REPO_DIR:-$PWD}"
python_cmd="${CIPT_PYTHON:-python}"
legacy_checkpoint="${CIPT_LEGACY_CHECKPOINT:-/home/hooasia/lgg/projects/repro_dccl_3domain/DCCL/DCCL/train_output/cipt_mechanism/PACS/261023_14-43-14_cipt_original_te0_seed0/checkpoints/TE0_5000.pth}"
upstream_checkpoint="${CIPT_UPSTREAM_CHECKPOINT:-$repo_dir/train_output/ablation/cipt_official_lr5e5_te0/PACS/target_0_art_painting/seed_0_trial_0/final_trainable.pth}"
data_dir="${CIPT_DATA_DIR:-/home/hooasia/lgg/data/repro_dccl_data}"
clip_path="${CIPT_CLIP_PATH:-/home/hooasia/.cache/clip/ViT-B-16.pt}"
review_root="${CIPT_REVIEW_OUTPUT:-$repo_dir/train_output/ablation/cipt_pair_review_$(date +%Y%m%d_%H%M%S)_$BASHPID}"

[[ -f "$repo_dir/domainbed/algorithms/cipt_env_kl.py" ]] || {
    echo 'Run this script from the training repository DCCL/DCCL directory.' >&2
    exit 1
}
for checkpoint in "$legacy_checkpoint" "$upstream_checkpoint" "$clip_path"; do
    [[ -f "$checkpoint" ]] || { printf 'Missing file: %s\n' "$checkpoint" >&2; exit 1; }
done
resume_args=()
if [[ "${CIPT_REVIEW_RESUME:-0}" == 1 ]]; then
    [[ -f "$review_root/protocol.json" ]] || { printf 'No saved protocol: %s\n' "$review_root" >&2; exit 1; }
    [[ ! -f "$review_root/status.json" ]] || { printf 'Already complete: %s\n' "$review_root"; exit 0; }
    resume_args+=(--resume)
else
    [[ ! -e "$review_root" ]] || { printf 'Output already exists: %s\n' "$review_root" >&2; exit 1; }
fi
mkdir -p -- "$(dirname -- "$review_root")"
export CUDA_VISIBLE_DEVICES="${CIPT_GPU:-1}"
export PYTHONUNBUFFERED=1

printf 'OUTPUT_DIR=%s\nCUDA_VISIBLE_DEVICES=%s\n' "$review_root" "$CUDA_VISIBLE_DEVICES"
"$python_cmd" -u "$script_dir/audit_cipt_pair.py" \
    --repo-root "$repo_dir" \
    --legacy-checkpoint "$legacy_checkpoint" \
    --upstream-checkpoint "$upstream_checkpoint" \
    --data-dir "$data_dir" --clip-path "$clip_path" \
    --batch-size 64 --num-workers 0 \
    "${resume_args[@]}" \
    --output-dir "$review_root" \
    2>&1 | tee -a "$review_root.console.log"

mv -- "$review_root.console.log" "$review_root/console.log"
cp -- "$script_dir/audit_cipt_pair.py" "$review_root/audit_cipt_pair.py"
cp -- "$script_dir/run_pair_gpu1.sh" "$review_root/launcher.sh"
printf '\nCOMPLETE: %s\nRead: %s/comparison.md\n' "$review_root" "$review_root"
