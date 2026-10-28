#!/usr/bin/env bash
# GPU0, serial six-way PACS / VLCS B5b center-vs-RD ablation.
# All runs: E=1, S=1, env constraint weight=1, IND=0, contrastive=0.
# Data, Python, CLIP and result root can be overridden by environment variables.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."

expected_branch="feature/cipt-b5b-entropy-mcc-rd"
[[ "$(git branch --show-current)" == "$expected_branch" ]] || {
    echo "ERROR: switch to branch $expected_branch before running." >&2
    exit 1
}
data_dir="${CIPT_DATA_DIR:-/home/hooasia/lgg/data/repro_dccl_data}"
clip_path="${CIPT_CLIP_PATH:-/home/hooasia/.cache/clip/ViT-B-16.pt}"
python_cmd="${CIPT_PYTHON:-$(command -v python)}"
base_root="${CIPT_OUTPUT_ROOT:-train_output/ablation/pacs_vlcs_center_rd_gpu0_serial}"
[[ -d "$data_dir" ]] || { echo "ERROR: dataset directory missing: $data_dir" >&2; exit 1; }
[[ -f "$clip_path" ]] || { echo "ERROR: CLIP checkpoint missing: $clip_path" >&2; exit 1; }
[[ -n "$python_cmd" && -x "$python_cmd" ]] || { echo "ERROR: Python executable missing: $python_cmd" >&2; exit 1; }
for cfg in config_cipt_b5b_center_mse.yaml config_cipt_b5b_all80_center_mse.yaml config_cipt_b5b_all80_relative_distance.yaml; do
    [[ -f "$cfg" ]] || { echo "ERROR: config missing: $cfg" >&2; exit 1; }
done
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i 0 --query-gpu=index,name --format=csv,noheader >/dev/null || {
        echo "ERROR: physical GPU 0 inaccessible." >&2
        exit 1
    }
fi

mkdir -p -- "$base_root"
run_root="$(mktemp -d -- "$base_root/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
cp -- "$script_dir/$(basename -- "$0")" "$run_root/launcher_snapshot.sh"
for cfg in config_cipt_b5b_center_mse.yaml config_cipt_b5b_all80_center_mse.yaml config_cipt_b5b_all80_relative_distance.yaml; do
    cp -- "$cfg" "$run_root/$cfg"
done
git rev-parse HEAD > "$run_root/git_commit.txt"
printf 'id\tdataset\tselection\ttopk\tconstraint\tE_weight\tS_weight\tenv_weight\tstatus\toutput_root\n' > "$run_root/status.tsv"
export PYTHONUNBUFFERED=1

# Order is intentional: do not run experiments in parallel.
# id|dataset|selection|topk|constraint|config
experiments=(
    '01|PACS|entropy|40|center_mse|config_cipt_b5b_center_mse.yaml'
    '02|VLCS|all|80|relative_distance|config_cipt_b5b_all80_relative_distance.yaml'
    '03|VLCS|all|80|center_mse|config_cipt_b5b_all80_center_mse.yaml'
    '04|VLCS|entropy|40|center_mse|config_cipt_b5b_center_mse.yaml'
    '05|PACS|random|40|center_mse|config_cipt_b5b_center_mse.yaml'
    '06|VLCS|random|40|center_mse|config_cipt_b5b_center_mse.yaml'
)

printf 'RUN_ROOT=%s\nGPU=0 (physical GPU0)\n' "$run_root"
printf 'PACS + VLCS: all 4 held-out domains each; official test_out, no --test_envs override.\n'
printf 'Fixed: E=1, S=1, env=1, IND=0, contrastive=0, seed=0, 5001 steps, batch=32, lr=5e-5, SWAD=LossValley.\n'

for spec in "${experiments[@]}"; do
    IFS='|' read -r id dataset selection topk constraint cfg <<< "$spec"
    name="${dataset,,}_${selection}${topk}_${constraint}_e1_s1_env1_swad"
    exp_root="$run_root/${id}_$name"
    mkdir -p -- "$exp_root"

    cmd=(
        "$python_cmd" -u train_all.py "$name" "$cfg"
        --algorithm CIPTDCCL --dataset "$dataset"
        --data_dir "$data_dir" --cipt_clip_path "$clip_path"
        --output_root "$exp_root"
        --deterministic --trial_seed 0 --seed 0
        --steps 5001 --checkpoint_freq 100
        --aug 0 --lr 5e-5 --batch_size 32
        --num_workers "${CIPT_NUM_WORKERS:-2}"
        --eval_num_workers "${CIPT_EVAL_NUM_WORKERS:-0}"
        --cipt_prompt_length 16 --cipt_prompt_init "a photo of a"
        --cipt_objective env_anchor --cipt_env_constraint "$constraint"
        --cipt_template_mode b5b --cipt_selector_mode random
        --cipt_env_selection "$selection" --cipt_env_topk "$topk"
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
        printf 'CUDA_VISIBLE_DEVICES=0 '
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "$exp_root/command.sh"

    printf '\n========== [%s/6] START %s at %s ==========\n' "$id" "$name" "$(date -Is)"
    if CUDA_VISIBLE_DEVICES=0 "${cmd[@]}" 2>&1 | tee "$exp_root/console.log"; then
        printf '%s\t%s\t%s\t%s\t%s\t1\t1\t1\tSUCCESS\t%s\n' \
            "$id" "$dataset" "$selection" "$topk" "$constraint" "$exp_root" >> "$run_root/status.tsv"
        printf '========== [%s/6] SUCCESS at %s ==========\n' "$id" "$(date -Is)"
    else
        status=$?
        printf '%s\t%s\t%s\t%s\t%s\t1\t1\t1\tFAILED(%s)\t%s\n' \
            "$id" "$dataset" "$selection" "$topk" "$constraint" "$status" "$exp_root" >> "$run_root/status.tsv"
        printf 'ERROR: experiment %s failed (status %s). Remaining experiments are NOT started.\n' "$id" "$status" >&2
        printf 'Inspect: %s/console.log\n' "$exp_root" >&2
        exit "$status"
    fi
done

echo "ALL SIX EXPERIMENTS COMPLETED: $run_root"
cat "$run_root/status.tsv"
