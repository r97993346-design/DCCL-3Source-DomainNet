#!/usr/bin/env bash
# Six VLCS leave-one-domain-out ablations, serially on physical GPU 0.
# Run from the activated DCCL environment. No target-domain labels enter training.
# One train_all.py invocation trains/evaluates all four held-out target domains.
set -Eeuo pipefail

dry_run=false
start_tmux=false
for option in "$@"; do
    case "$option" in
        --dry-run) dry_run=true ;;
        --tmux) start_tmux=true ;;
        -h|--help)
            printf 'Usage: bash %s [--dry-run | --tmux]\n' "$0"
            printf 'Overrides: CIPT_PROJECT_DIR, CIPT_DATA_DIR, CIPT_CLIP_PATH, CIPT_PYTHON, CIPT_SWEEP_OUTPUT_ROOT\n'
            exit 0 ;;
        *) printf 'ERROR: unknown option %s\n' "$option" >&2; exit 2 ;;
    esac
done
if $dry_run && $start_tmux; then
    echo 'ERROR: use --dry-run or --tmux separately.' >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
script_path="$script_dir/$(basename -- "${BASH_SOURCE[0]}")"
if [[ -n "${CIPT_PROJECT_DIR:-}" ]]; then
    project_dir="$CIPT_PROJECT_DIR"
elif [[ -f "$script_dir/../train_all.py" ]]; then
    project_dir="$script_dir/.."
elif [[ -f ./train_all.py ]]; then
    project_dir="$PWD"
else
    project_dir="/home/hooasia/lgg/projects/repro_dccl_3domain/DCCL/DCCL"
fi
cd -- "$project_dir"
project_dir="$PWD"

expected_branch="feature/cipt-b5b-entropy-mcc-rd"
data_dir="${CIPT_DATA_DIR:-/home/hooasia/lgg/data/repro_dccl_data}"
clip_path="${CIPT_CLIP_PATH:-/home/hooasia/.cache/clip/ViT-B-16.pt}"
python_cmd="$(command -v "${CIPT_PYTHON:-python}")"
base_root="${CIPT_SWEEP_OUTPUT_ROOT:-train_output/ablation/vlcs_ablation6_gpu0}"

if ! $dry_run; then
    current_branch="$(git branch --show-current)"
    [[ "$current_branch" == "$expected_branch" ]] || {
        printf 'ERROR: expected branch %s, found %s\n' "$expected_branch" "$current_branch" >&2
        exit 1
    }
    [[ -f train_all.py && -f config.yaml &&
       -f config_cipt_clip_ce.yaml && -f config_cipt_b5b_entropy_rd.yaml ]] || {
        echo 'ERROR: missing train_all.py or experiment YAML files.' >&2
        exit 1
    }
    [[ -d "$data_dir/VLCS" ]] || { echo "ERROR: VLCS directory not found: $data_dir/VLCS" >&2; exit 1; }
    [[ -f "$clip_path" ]] || { echo "ERROR: CLIP checkpoint not found: $clip_path" >&2; exit 1; }
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -i 0 --query-gpu=index,name --format=csv,noheader >/dev/null || {
            echo 'ERROR: GPU0 is not accessible.' >&2
            exit 1
        }
    fi
fi

# Capture the active environment's absolute Python path for a new tmux shell.
if $start_tmux; then
    command -v tmux >/dev/null || { echo 'ERROR: tmux is not installed.' >&2; exit 1; }
    session="vlcs_ablation6_gpu0_$(date +%Y%m%d_%H%M%S)_$$"
    printf -v launch 'CIPT_PROJECT_DIR=%q CIPT_DATA_DIR=%q CIPT_CLIP_PATH=%q CIPT_PYTHON=%q CIPT_SWEEP_OUTPUT_ROOT=%q bash %q' \
        "$project_dir" "$data_dir" "$clip_path" "$python_cmd" "$base_root" "$script_path"
    tmux new-session -d -s "$session" -c "$project_dir"
    tmux send-keys -t "$session" -l -- "$launch"
    tmux send-keys -t "$session" Enter
    printf 'Started serial VLCS experiments on GPU0.\nAttach: tmux attach -t %s\n' "$session"
    exit 0
fi

export PYTHONUNBUFFERED=1
if $dry_run; then
    run_root="$base_root/DRY_RUN"
else
    mkdir -p -- "$base_root"
    run_root="$(mktemp -d -- "$base_root/sweep_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
    cp -- "$script_path" "$run_root/launcher_snapshot.sh"
    cp -- config.yaml config_cipt_clip_ce.yaml config_cipt_b5b_entropy_rd.yaml "$run_root/"
    git rev-parse HEAD > "$run_root/git_commit.txt"
    printf 'experiment\tobjective\tcontrastive\ttemperature\tcontrastive_weight\tRD_weight\tstatus\toutput_root\n' > "$run_root/status.tsv"
fi

printf 'SWEEP_ROOT=%s\nGPU=0; dataset=VLCS; seed=trial_seed=0; steps=5001; SWAD=LossValley\n' "$run_root"
printf 'Groups 2-6 classify E. RD uses entropy top40 and the full 80-template class center.\n'
printf 'Each group covers TE0, TE1, TE2, TE3 automatically; all 24 training runs are serial.\n'

# name | objective | contrastive type | temperature | contrastive weight | RD weight
# Group 5 is E/S + RD only; group 6 combines the group 4 and group 5 modules.
experiments=(
    '01_clip_ce|clip_ce|supcon|0.1|0.0|0.0'
    '02_e1_s1|env_anchor|supcon|0.07|0.0|0.0'
    '03_e1_s1_supcon_t01_w1|env_anchor|supcon|0.1|1.0|0.0'
    '04_e1_s1_neighbor_t007_w1p5|env_anchor|neighbor_retention|0.07|1.5|0.0'
    '05_e1_s1_rd2_n40_full|env_anchor|supcon|0.07|0.0|2.0'
    '06_e1_s1_neighbor_rd2_n40_full|env_anchor|neighbor_retention|0.07|1.5|2.0'
)

for spec in "${experiments[@]}"; do
    IFS='|' read -r run_name objective mode tau con_weight rd_weight <<< "$spec"
    config_file=config_cipt_b5b_entropy_rd.yaml
    use_de=true
    s_weight=1.0
    pure=false
    if [[ "$objective" == clip_ce ]]; then
        config_file=config_cipt_clip_ce.yaml
        use_de=false
        s_weight=0.0
        pure=true
    fi
    use_contrastive=false
    [[ "$con_weight" == 0.0 ]] || use_contrastive=true
    env_constraint=none
    [[ "$rd_weight" == 0.0 ]] || env_constraint=relative_distance

    exp_root="$run_root/$run_name"
    cmd=(
        "$python_cmd" -u train_all.py "$run_name" "$config_file"
        --algorithm CIPTDCCL --dataset VLCS
        --data_dir "$data_dir" --cipt_clip_path "$clip_path"
        --output_root "$exp_root"
        --steps 5001 --checkpoint_freq 100
        --deterministic --seed 0 --trial_seed 0 --holdout_fraction 0.2
        --aug 0 --lr 5e-5 --batch_size 32 --swad LossValley --model_save 0
        --cipt_clip_backbone ViT-B/16
        --cipt_prompt_length 16 --cipt_prompt_init "a photo of a"
        --cipt_objective "$objective"
        --cipt_use_de "$use_de" --cipt_beta 1.0
        --cipt_e_cls_weight 1.0 --cipt_s_uniform_weight "$s_weight"
        --cipt_use_ind false --cipt_gamma 0.0
        --cipt_use_tda false --cipt_env_inference e_only
        --cipt_env_kl_weight 0.0 --cipt_fused_cls_weight 0.0
        --cipt_pure "$pure" --cipt_use_aug_view false
        --cipt_template_mode b5b --cipt_selector_mode random
        --cipt_env_constraint "$env_constraint" --cipt_env_loss_weight "$rd_weight"
        --cipt_env_selection entropy --cipt_env_topk 40 --cipt_rd_anchor_mode full
        --cipt_use_contrastive "$use_contrastive" --cipt_contrastive_type "$mode"
        --cipt_contrastive_temperature "$tau"
        --cipt_causal_contrastive_weight "$con_weight"
        --cipt_contrastive_warmup_steps 0
        --cipt_neighbor_k 10 --cipt_neighbor_alpha 4.0
    )

    if $dry_run; then
        printf '\nCUDA_VISIBLE_DEVICES=0 '
        printf '%q ' "${cmd[@]}"
        printf '\n'
        continue
    fi

    mkdir -- "$exp_root"
    printf '#!/usr/bin/env bash\ncd -- %q\nCUDA_VISIBLE_DEVICES=0 ' "$project_dir" > "$exp_root/command.sh"
    printf '%q ' "${cmd[@]}" >> "$exp_root/command.sh"
    printf '\n' >> "$exp_root/command.sh"
    printf '\n========== START %s (%s) ==========\n' "$run_name" "$(date -Is)"
    if CUDA_VISIBLE_DEVICES=0 "${cmd[@]}" 2>&1 | tee "$exp_root/console.log"; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\tSUCCESS\t%s\n' \
            "$run_name" "$objective" "$mode" "$tau" "$con_weight" "$rd_weight" "$exp_root" >> "$run_root/status.tsv"
        printf '========== SUCCESS %s (%s) ==========\n' "$run_name" "$(date -Is)"
    else
        code=$?
        printf '%s\t%s\t%s\t%s\t%s\t%s\tFAILED(%s)\t%s\n' \
            "$run_name" "$objective" "$mode" "$tau" "$con_weight" "$rd_weight" "$code" "$exp_root" >> "$run_root/status.tsv"
        printf 'ERROR: %s failed (exit %s); remaining experiments stopped. See %s/console.log\n' \
            "$run_name" "$code" "$exp_root" >&2
        exit "$code"
    fi
done

if ! $dry_run; then
    printf '\nALL SIX EXPERIMENTS COMPLETED: %s\n' "$run_root"
    cat -- "$run_root/status.tsv"
fi
