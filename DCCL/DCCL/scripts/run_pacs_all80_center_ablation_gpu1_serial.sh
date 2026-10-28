#!/usr/bin/env bash
# GPU1 serial, independent ablations around PACS all80 center_mse E1 S1 env1 baseline.
# 1) center=2; 2) center=.5; 3) IND=1 (center=1); 4) NR contrast=1 temp=.1 warmup=0 (center=1).
# Each job trains all 4 PACS target domains; IID evaluation is in the regular log.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
expected_branch="feature/cipt-b5b-entropy-mcc-rd"
if [[ "$(git branch --show-current)" != "$expected_branch" ]]; then
    echo "ERROR: switch to $expected_branch" >&2
    exit 1
fi

data_dir="${CIPT_DATA_DIR:-/home/hooasia/lgg/data/repro_dccl_data}"
clip_path="${CIPT_CLIP_PATH:-/home/hooasia/.cache/clip/ViT-B-16.pt}"
python_cmd="${CIPT_PYTHON:-$(command -v python)}"
base_root="${CIPT_OUTPUT_ROOT:-train_output/ablation/pacs_all80_center_ablation_gpu1_serial}"
[[ -d "$data_dir" ]] || { echo "ERROR: no dataset at $data_dir" >&2; exit 1; }
[[ -f "$clip_path" ]] || { echo "ERROR: no CLIP model at $clip_path" >&2; exit 1; }
[[ -n "$python_cmd" && -x "$python_cmd" ]] || { echo "ERROR: python not executable: $python_cmd" >&2; exit 1; }
[[ -f config_cipt_b5b_all80_center_mse.yaml ]] || { echo "ERROR: center config missing" >&2; exit 1; }
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i 1 --query-gpu=index,name --format=csv,noheader >/dev/null || {
        echo "ERROR: physical GPU1 is inaccessible" >&2
        exit 1
    }
fi

mkdir -p -- "$base_root"
run_root="$(mktemp -d -- "$base_root/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
cp -- "$script_dir/$(basename -- "$0")" "$run_root/launcher_snapshot.sh"
cp -- config_cipt_b5b_all80_center_mse.yaml "$run_root/"
git rev-parse HEAD > "$run_root/git_commit.txt"
printf 'id\trun_name\tcenter_weight\tind_weight\tcontrastive_type\tcontrastive_weight\ttemperature\twarmup\tstatus\toutput_root\n' > "$run_root/status.tsv"
export PYTHONUNBUFFERED=1

# Each is independent relative to the exact E1/S1/center1/IND0/contrast0 reference.
# id|shortname|center_weight|ind_weight|contrastive_type|contrastive_weight|temp|warmup
experiments=(
    '01|center2|2.0|0.0|neighbor_retention|0.0|0.1|500'
    '02|center05|0.5|0.0|neighbor_retention|0.0|0.1|500'
    '03|ind1|1.0|1.0|neighbor_retention|0.0|0.1|500'
    '04|nr_temp01_warmup0_con1|1.0|0.0|neighbor_retention|1.0|0.1|0'
)

echo "RUN_ROOT=$run_root"
echo "GPU=1; PACS four held-out domains each; fixed E=1 S=1 all80 Center MSE, lr=5e-5, batch=32, seed=0, steps=5001, SWAD LossValley."
echo "Run 3 isolates IND=1, run 4 isolates NR contrastive; both retain Center MSE=1."

for spec in "${experiments[@]}"; do
    IFS='|' read -r id short center_w ind_w con_type con_w temp warmup <<< "$spec"
    run_name="pacs_all80_center_e1_s1_${short}_swad"
    exp_root="$run_root/${id}_$run_name"
    mkdir -p -- "$exp_root"

    if [[ "$ind_w" == "0.0" ]]; then
        use_ind=false
    else
        use_ind=true
    fi
    if [[ "$con_w" == "0.0" ]]; then
        use_contrastive=false
    else
        use_contrastive=true
    fi
    cmd=(
        "$python_cmd" -u train_all.py "$run_name" config_cipt_b5b_all80_center_mse.yaml
        --algorithm CIPTDCCL --dataset PACS
        --data_dir "$data_dir" --cipt_clip_path "$clip_path"
        --output_root "$exp_root"
        --deterministic --trial_seed 0 --seed 0
        --steps 5001 --checkpoint_freq 100
        --aug 0 --lr 5e-5 --batch_size 32
        --num_workers "${CIPT_NUM_WORKERS:-2}"
        --eval_num_workers "${CIPT_EVAL_NUM_WORKERS:-0}"
        --cipt_prompt_length 16 --cipt_prompt_init "a photo of a"
        --cipt_objective env_anchor --cipt_env_constraint center_mse
        --cipt_template_mode b5b --cipt_selector_mode random
        --cipt_env_selection all --cipt_env_topk 80
        --cipt_rd_anchor_mode full --cipt_env_loss_weight "$center_w"
        --cipt_use_de true --cipt_beta 1.0
        --cipt_e_cls_weight 1.0 --cipt_s_uniform_weight 1.0
        --cipt_use_ind "$use_ind" --cipt_gamma "$ind_w"
        --cipt_use_tda false --cipt_env_inference e_only
        --cipt_env_kl_weight 0.0 --cipt_fused_cls_weight 0.0
        --cipt_pure false --cipt_use_aug_view false
        --cipt_use_contrastive "$use_contrastive"
        --cipt_contrastive_type "$con_type"
        --cipt_contrastive_temperature "$temp"
        --cipt_contrastive_warmup_steps "$warmup"
        --cipt_causal_contrastive_weight "$con_w"
        --swad LossValley
    )
    {
        printf 'CUDA_VISIBLE_DEVICES=1 '
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "$exp_root/command.sh"
    printf '\n========== [%s/4] START %s at %s ==========\n' "$id" "$run_name" "$(date -Is)"
    if CUDA_VISIBLE_DEVICES=1 "${cmd[@]}" 2>&1 | tee "$exp_root/console.log"; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tSUCCESS\t%s\n' \
            "$id" "$run_name" "$center_w" "$ind_w" "$con_type" "$con_w" "$temp" "$warmup" "$exp_root" >> "$run_root/status.tsv"
        printf '========== [%s/4] SUCCESS at %s ==========\n' "$id" "$(date -Is)"
    else
        code=$?
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tFAILED(%s)\t%s\n' \
            "$id" "$run_name" "$center_w" "$ind_w" "$con_type" "$con_w" "$temp" "$warmup" "$code" "$exp_root" >> "$run_root/status.tsv"
        printf 'ERROR: experiment %s exited with %s; remaining jobs will NOT start.\n' "$id" "$code" >&2
        printf 'Log: %s/console.log\n' "$exp_root" >&2
        exit "$code"
    fi
done

printf '\nALL FOUR EXPERIMENTS COMPLETED. Results: %s\n' "$run_root"
cat "$run_root/status.tsv"
