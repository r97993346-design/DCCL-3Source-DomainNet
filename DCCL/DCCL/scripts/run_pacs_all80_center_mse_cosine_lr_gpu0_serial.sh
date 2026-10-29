#!/usr/bin/env bash
# PACS all80 Center MSE, E=1 S=1 env-weight=1, Adam + cosine LR.
# GPU0 serial order: 0.0025 -> 0.002 -> 0.0001 -> 5e-5.
# Each run trains ALL FOUR PACS held-out environments and logs IID/SWAD.
# This launcher does not overwrite earlier results or run jobs in parallel.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."

expected_branch="feature/cipt-b5b-entropy-mcc-rd"
if [[ "$(git branch --show-current)" != "$expected_branch" ]]; then
    printf 'ERROR: expected Git branch %s\n' "$expected_branch" >&2
    exit 1
fi

data_dir="${CIPT_DATA_DIR:-/home/hooasia/lgg/data/repro_dccl_data}"
clip_path="${CIPT_CLIP_PATH:-/home/hooasia/.cache/clip/ViT-B-16.pt}"
python_cmd="${CIPT_PYTHON:-$(command -v python)}"
base_root="${CIPT_OUTPUT_ROOT:-train_output/ablation/pacs_all80_center_mse_cosine_lr_gpu0_serial}"

[[ -d "$data_dir" ]] || { echo "ERROR: PACS dataset directory missing: $data_dir" >&2; exit 1; }
[[ -f "$clip_path" ]] || { echo "ERROR: CLIP checkpoint missing: $clip_path" >&2; exit 1; }
[[ -n "$python_cmd" && -x "$python_cmd" ]] || { echo "ERROR: invalid Python executable: $python_cmd" >&2; exit 1; }
for cfg in config_cipt_b5b_all80_center_mse.yaml config_cipt_lr_cosine.yaml; do
    [[ -f "$cfg" ]] || { echo "ERROR: missing config: $cfg" >&2; exit 1; }
done
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i 0 --query-gpu=index,name --format=csv,noheader >/dev/null || {
        echo "ERROR: physical GPU0 is not accessible" >&2
        exit 1
    }
fi

mkdir -p -- "$base_root"
run_root="$(mktemp -d -- "$base_root/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
cp -- "$script_dir/$(basename -- "$0")" "$run_root/launcher_snapshot.sh"
cp -- config_cipt_b5b_all80_center_mse.yaml config_cipt_lr_cosine.yaml "$run_root/"
git rev-parse HEAD > "$run_root/git_commit.txt"
printf 'order\tinitial_lr\tschedule\tsteps_arg\tcheckpoint_freq\tstatus\toutput_root\n' > "$run_root/status.tsv"

export PYTHONUNBUFFERED=1

# id|initial_lr|short_name (intentional experiment order)
experiments=(
    '01|0.0025|lr0025'
    '02|0.002|lr002'
    '03|0.0001|lr0001'
    '04|5e-5|lr5e5'
)

printf '\nRUN_ROOT=%s\nGPU0 serial, PACS 4 target domains per run.\n' "$run_root"
echo 'Common: all80 B5b + Center MSE (weight=1), E=1 S=1, IND=0, contrast=0, learnable class text.'
echo 'Adam cosine annealing: each run starts at its specified initial LR, decays to 0, T_max=5001 update steps.'
echo 'train_all.py rounds --steps 5000 with checkpoint_freq 100 to 5001 optimizer updates.'

for spec in "${experiments[@]}"; do
    IFS='|' read -r id initial_lr short <<< "$spec"
    exp_name="pacs_all80_center_mse_e1_s1_env1_${short}_cosine_5000steps_swad"
    exp_root="$run_root/${id}_$exp_name"
    mkdir -p -- "$exp_root"

    cmd=(
        "$python_cmd" -u train_all.py "$exp_name"
        config_cipt_b5b_all80_center_mse.yaml config_cipt_lr_cosine.yaml
        --algorithm CIPTDCCL --dataset PACS
        --data_dir "$data_dir" --cipt_clip_path "$clip_path"
        --output_root "$exp_root"
        --deterministic --trial_seed 0 --seed 0
        --steps 5000 --checkpoint_freq 100
        --aug 0 --lr "$initial_lr" --batch_size 32
        --num_workers "${CIPT_NUM_WORKERS:-2}"
        --eval_num_workers "${CIPT_EVAL_NUM_WORKERS:-0}"
        --cipt_prompt_length 16 --cipt_prompt_init "a photo of a"
        --cipt_objective env_anchor --cipt_class_text_mode learnable
        --cipt_env_constraint center_mse
        --cipt_template_mode b5b --cipt_selector_mode random
        --cipt_env_selection all --cipt_env_topk 80
        --cipt_rd_anchor_mode full --cipt_env_loss_weight 1.0
        --cipt_lr_schedule cosine --cipt_lr_min 0.0
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
        printf 'CUDA_VISIBLE_DEVICES=0 '
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "$exp_root/command.sh"

    printf '\n========== [%s/4] START lr=%s at %s ==========\n' "$id" "$initial_lr" "$(date -Is)"
    if CUDA_VISIBLE_DEVICES=0 "${cmd[@]}" 2>&1 | tee "$exp_root/console.log"; then
        printf '%s\t%s\tcosine\t5000\t100\tSUCCESS\t%s\n' \
            "$id" "$initial_lr" "$exp_root" >> "$run_root/status.tsv"
        printf '========== [%s/4] COMPLETE lr=%s at %s ==========\n' "$id" "$initial_lr" "$(date -Is)"
    else
        status=$?
        printf '%s\t%s\tcosine\t5000\t100\tFAILED(%s)\t%s\n' \
            "$id" "$initial_lr" "$status" "$exp_root" >> "$run_root/status.tsv"
        echo "ERROR: lr=$initial_lr failed with exit status $status; remaining runs will NOT start." >&2
        echo "Inspect: $exp_root/console.log" >&2
        exit "$status"
    fi
done

printf '\nALL FOUR RUNS FINISHED: %s\n' "$run_root"
cat "$run_root/status.tsv"
