#!/usr/bin/env bash
# Ten controlled PACS leave-one-domain-out experiments, serially on GPU 1.
# Run inside the already activated (dccl) environment.
# Each invocation allocates a new unique output root; existing runs are untouched.
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
python_cmd="${CIPT_PYTHON:-python}"
base_root="${CIPT_SWEEP_OUTPUT_ROOT:-train_output/ablation/rd2_full_anchor_gpu1_sweep10}"

[[ -d "$data_dir" ]] || { echo "ERROR: dataset not found: $data_dir" >&2; exit 1; }
[[ -f "$clip_path" ]] || { echo "ERROR: CLIP checkpoint not found: $clip_path" >&2; exit 1; }
[[ -f config_cipt_b5b_entropy_rd.yaml ]] || { echo "ERROR: missing RD YAML" >&2; exit 1; }
command -v "$python_cmd" >/dev/null || { echo "ERROR: Python not found: $python_cmd" >&2; exit 1; }
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i 1 --query-gpu=index,name --format=csv,noheader >/dev/null ||
        { echo "ERROR: GPU1 is not accessible" >&2; exit 1; }
fi

mkdir -p -- "$base_root"
run_root="$(mktemp -d -- "$base_root/sweep_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
cp -- "$script_dir/$(basename -- "$0")" "$run_root/launcher_snapshot.sh"
cp -- config_cipt_b5b_entropy_rd.yaml "$run_root/config_snapshot.yaml"
git rev-parse HEAD > "$run_root/git_commit.txt"
printf 'index\tmode\ttau\tweight\tind\tstatus\toutput_root\n' > "$run_root/status.tsv"
export PYTHONUNBUFFERED=1

printf 'SWEEP_ROOT=%s\nGPU=1\nRD_anchor=full\nRD_weight=2.0\nEntropy_topk=40\n' "$run_root"
printf 'There are 10 serial runs; each train_all.py call covers all four PACS target domains.\n'
printf 'DataLoader workers: train=%s per source domain; eval=%s per loader.\n' \
    "${CIPT_NUM_WORKERS:-2}" "${CIPT_EVAL_NUM_WORKERS:-0}"

# id | contrastive_type | E contrastive temperature | contrastive weight | IND (gamma=1 when enabled)
experiments=(
    '01|supcon|0.07|1.5|0'
    '02|clip_guided_cross_domain|0.07|1.5|0'
    '03|neighbor_retention|0.07|2.0|0'
    '04|neighbor_retention|0.07|1.5|1'
    '05|supcon|0.1|1.5|0'
    '06|neighbor_retention|0.1|1.5|0'
    '07|supcon|0.1|2.0|0'
    '08|neighbor_retention|0.1|2.0|0'
    '09|clip_guided_cross_domain|0.1|1.5|0'
    '10|clip_guided_cross_domain|0.1|2.0|0'
)

for spec in "${experiments[@]}"; do
    IFS='|' read -r id mode tau weight ind <<< "$spec"
    if [[ "$ind" == "1" ]]; then
        use_ind=true
        gamma=1.0
    else
        use_ind=false
        gamma=0.0
    fi

    run_name="exp${id}_${mode}_tau${tau//./p}_w${weight//./p}_ind${ind}"
    exp_root="$run_root/$run_name"
    mkdir -- "$exp_root"

    cmd=(
        "$python_cmd" train_all.py "$run_name" config_cipt_b5b_entropy_rd.yaml
        --algorithm CIPTDCCL --dataset PACS
        --data_dir "$data_dir"
        --cipt_clip_path "$clip_path"
        --output_root "$exp_root"
        --deterministic --seed 0 --trial_seed 0
        --aug 0 --lr 5e-5 --batch_size 32 --checkpoint_freq 100
        --num_workers "${CIPT_NUM_WORKERS:-2}"
        --eval_num_workers "${CIPT_EVAL_NUM_WORKERS:-0}"
        --cipt_prompt_length 16 --cipt_prompt_init "a photo of a"
        --cipt_objective env_anchor --cipt_env_constraint relative_distance
        --cipt_template_mode b5b --cipt_selector_mode random
        --cipt_env_selection entropy --cipt_env_topk 40
        --cipt_rd_anchor_mode full --cipt_env_loss_weight 2.0
        --cipt_use_de true --cipt_beta 1.0
        --cipt_e_cls_weight 1.0 --cipt_s_uniform_weight 1.0
        --cipt_use_ind "$use_ind" --cipt_gamma "$gamma"
        --cipt_use_tda false --cipt_env_inference e_only
        --cipt_env_kl_weight 0.0 --cipt_fused_cls_weight 0.0
        --cipt_pure false --cipt_use_aug_view false
        --cipt_use_contrastive true --cipt_contrastive_type "$mode"
        --cipt_neighbor_k 10 --cipt_neighbor_alpha 4.0
        --cipt_contrastive_temperature "$tau"
        --cipt_causal_contrastive_weight "$weight"
        --cipt_contrastive_warmup_steps 0
        --cipt_clip_teacher_temperature 0.2
        --cipt_clip_teacher_mix 0.5
        --cipt_clip_teacher_topk 0
        --swad LossValley
    )

    printf 'CUDA_VISIBLE_DEVICES=1 ' > "$exp_root/command.sh"
    printf '%q ' "${cmd[@]}" >> "$exp_root/command.sh"
    printf '\n' >> "$exp_root/command.sh"

    printf '\n========== [%s/10] START %s (%s) ==========\n' "$id" "$run_name" "$(date -Is)"
    if CUDA_VISIBLE_DEVICES=1 "${cmd[@]}" 2>&1 | tee "$exp_root/console.log"; then
        printf '%s\t%s\t%s\t%s\t%s\tSUCCESS\t%s\n' \
            "$id" "$mode" "$tau" "$weight" "$ind" "$exp_root" >> "$run_root/status.tsv"
        printf '========== [%s/10] SUCCESS (%s) ==========\n' "$id" "$(date -Is)"
    else
        code=$?
        printf '%s\t%s\t%s\t%s\t%s\tFAILED(%s)\t%s\n' \
            "$id" "$mode" "$tau" "$weight" "$ind" "$code" "$exp_root" >> "$run_root/status.tsv"
        printf 'ERROR: experiment %s failed (exit %s); subsequent runs will NOT start.\n' "$id" "$code" >&2
        exit "$code"
    fi
done

printf '\nALL 10 EXPERIMENTS COMPLETED: %s\n' "$run_root"
cat "$run_root/status.tsv"

