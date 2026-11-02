#!/usr/bin/env bash
# Serial PACS comparison of all-80 official B5b template constraints on physical GPU 1.
# Run from any directory after pulling this branch; logs/config snapshots are preserved.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${script_dir}/.."

expected_branch="feature/cipt-b5b-entropy-mcc-rd"
current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$expected_branch" ]]; then
    printf 'ERROR: expected branch %s, found %s\n' "$expected_branch" "$current_branch" >&2
    exit 1
fi

data_dir="${CIPT_DATA_DIR:-/home/hooasia/lgg/data/repro_dccl_data}"
clip_path="${CIPT_CLIP_PATH:-/home/hooasia/.cache/clip/ViT-B-16.pt}"
python_cmd="${CIPT_PYTHON:-/home/hooasia/miniconda3/envs/dccl/bin/python}"
base_root="${CIPT_OUTPUT_ROOT:-train_output/ablation/rd_forms_all80_gpu1}"

[[ -d "$data_dir" ]] || { printf 'ERROR: dataset not found: %s\n' "$data_dir" >&2; exit 1; }
[[ -f "$clip_path" ]] || { printf 'ERROR: CLIP checkpoint not found: %s\n' "$clip_path" >&2; exit 1; }
[[ -x "$python_cmd" ]] || { printf 'ERROR: Python is not executable: %s\n' "$python_cmd" >&2; exit 1; }
for cfg in \
    config_cipt_b5b_all80_center_mse.yaml \
    config_cipt_b5b_all80_relative_distance.yaml \
    config_cipt_b5b_all80_mean_template_mse.yaml; do
    [[ -f "$cfg" ]] || { printf 'ERROR: missing config: %s\n' "$cfg" >&2; exit 1; }
done
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i 1 --query-gpu=index,name --format=csv,noheader >/dev/null || {
        echo "ERROR: physical GPU 1 is not accessible." >&2
        exit 1
    }
fi

mkdir -p -- "$base_root"
run_root="$(mktemp -d -- "$base_root/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
cp -- "$script_dir/$(basename -- "$0")" "$run_root/launcher_snapshot.sh"
for cfg in \
    config_cipt_b5b_all80_center_mse.yaml \
    config_cipt_b5b_all80_relative_distance.yaml \
    config_cipt_b5b_all80_mean_template_mse.yaml; do
    cp -- "$cfg" "$run_root/$cfg"
done
git rev-parse HEAD > "$run_root/git_commit.txt"
printf 'id\tconstraint\tE_weight\tS_weight\tenv_weight\tstatus\toutput_root\n' > "$run_root/status.tsv"
export PYTHONUNBUFFERED=1

printf 'RUN_ROOT=%s\nGPU=1 (physical GPU 1)\n' "$run_root"
printf 'PACS: all four held-out target domains (no --test_envs override).\n'
printf 'Fixed: E=1, S=1, environment constraint weight=1, lr=5e-5, batch=32, seed=0, steps=5001.\n'
printf 'Official B5b text bank: all 80 templates; entropy/random selection bypassed.\n'

# Order: second formula (center), original third formula (RD), fourth formula (mean over templates).
experiments=(
    '01|center_mse|config_cipt_b5b_all80_center_mse.yaml'
    '02|relative_distance|config_cipt_b5b_all80_relative_distance.yaml'
    '03|mean_template_mse|config_cipt_b5b_all80_mean_template_mse.yaml'
)

for spec in "${experiments[@]}"; do
    IFS='|' read -r id constraint config <<< "$spec"
    run_name="pacs_all80_${constraint}_e1_s1_rd1_swad"
    exp_root="$run_root/$run_name"
    mkdir -p -- "$exp_root"

    cmd=(
        "$python_cmd" -u train_all.py "$run_name" "$config"
        --algorithm CIPTDCCL --dataset PACS
        --data_dir "$data_dir"
        --cipt_clip_path "$clip_path"
        --output_root "$exp_root"
        --deterministic --trial_seed 0 --seed 0
        --steps 5001 --checkpoint_freq 100
        --aug 0 --lr 5e-5 --batch_size 32
        --num_workers "${CIPT_NUM_WORKERS:-2}"
        --eval_num_workers "${CIPT_EVAL_NUM_WORKERS:-0}"
        --cipt_prompt_length 16 --cipt_prompt_init "a photo of a"
        --cipt_objective env_anchor --cipt_env_constraint "$constraint"
        --cipt_template_mode b5b --cipt_selector_mode random
        --cipt_env_selection all --cipt_env_topk 80
        --cipt_rd_anchor_mode full --cipt_env_loss_weight 1.0
        --cipt_use_de true --cipt_beta 1.0
        --cipt_e_cls_weight 1.0 --cipt_s_uniform_weight 1.0
        --cipt_use_ind false --cipt_gamma 0.0
        --cipt_use_tda false --cipt_env_inference e_only
        --cipt_env_kl_weight 0.0 --cipt_fused_cls_weight 0.0
        --cipt_pure false --cipt_use_aug_view false
        --cipt_use_contrastive false --cipt_causal_contrastive_weight 0.0
        --swad LossValley
    )

    {
        printf 'CUDA_VISIBLE_DEVICES=1 '
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "$exp_root/command.sh"

    printf '\n========== [%s/3] START %s at %s ==========\n' "$id" "$run_name" "$(date -Is)"
    if CUDA_VISIBLE_DEVICES=1 "${cmd[@]}" 2>&1 | tee "$exp_root/console.log"; then
        printf '%s\t%s\t1.0\t1.0\t1.0\tSUCCESS\t%s\n' \
            "$id" "$constraint" "$exp_root" >> "$run_root/status.tsv"
        printf '========== [%s/3] SUCCESS at %s ==========\n' "$id" "$(date -Is)"
    else
        code=$?
        printf '%s\t%s\t1.0\t1.0\t1.0\tFAILED(%s)\t%s\n' \
            "$id" "$constraint" "$code" "$exp_root" >> "$run_root/status.tsv"
        printf 'ERROR: experiment %s failed (exit %s); later experiments will not start.\n' "$id" "$code" >&2
        printf 'Inspect %s/console.log\n' "$exp_root" >&2
        exit "$code"
    fi
done

printf '\nALL THREE EXPERIMENTS COMPLETED. Results root: %s\n' "$run_root"
cat "$run_root/status.tsv"
