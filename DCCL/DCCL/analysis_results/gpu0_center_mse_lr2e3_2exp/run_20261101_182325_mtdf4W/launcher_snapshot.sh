#!/usr/bin/env bash
# PACS: two experiments, serially on physical GPU0.
# Activate the existing training environment before running this script.
# Fixed LR=2e-3, E=S=1, center_mse=1, IND=0; no contrastive -> NR=1.
set -Eeuo pipefail

usage() {
    cat <<'HELP'
Usage: bash run_cipt_gpu0_center_mse_lr2e3_2exp.sh [--tmux | --dry-run | --help]
  --tmux     Start the queue in a detached tmux session.
  --dry-run  Print the two commands without starting training.
  no option  Run in the current terminal.

1. center_mse=1, E=S=1, IND=0, LR=2e-3 constant, --steps 3000; contrastive OFF
2. Same settings, plus neighbor_retention=1, temperature=0.1, warmup=0

Shared: PACS targets 0,1,2,3; b5b all80; IND=0 (disabled);
        batch=32; seed=trial_seed=0; Adam; SWAD=LossValley; e_only.
NR experiment: neighbor k=10, alpha=4; weight=1; temperature=0.1; warmup=0.
Branch: feature/cipt-b5b-entropy-mcc-rd
Default repository: /home/hooasia/lgg/projects/cipt_gpu0
The folder name does not select the GPU; CUDA_VISIBLE_DEVICES=0 does.
The branch rounds --steps 3000 to 3001 optimizer updates.

Overrides: CIPT_TRAIN_DIR, CIPT_PYTHON, CIPT_DATA_DIR, CIPT_CLIP_PATH,
           CIPT_OUTPUT_PARENT, CIPT_TMUX_SESSION
The script checks the branch; it does not switch branches or pull code.
HELP
}

mode="${1:---run}"
if [[ $# -gt 1 ]]; then usage >&2; exit 2; fi
case "$mode" in
    --help|-h) usage; exit 0 ;;
    --run|--tmux|--dry-run) ;;
    *) usage >&2; exit 2 ;;
esac

script_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
repository_dir="/home/hooasia/lgg/projects/cipt_gpu0"
expected_branch="feature/cipt-b5b-entropy-mcc-rd"

if [[ -n "${CIPT_TRAIN_DIR:-}" ]]; then
    train_dir="$CIPT_TRAIN_DIR"
elif [[ -f "$repository_dir/DCCL/DCCL/train_all.py" ]]; then
    train_dir="$repository_dir/DCCL/DCCL"
elif [[ -f "$repository_dir/DCCL/train_all.py" ]]; then
    train_dir="$repository_dir/DCCL"
elif [[ -f "$repository_dir/train_all.py" ]]; then
    train_dir="$repository_dir"
else
    printf 'ERROR: train_all.py not found under %s. Set CIPT_TRAIN_DIR.\n' "$repository_dir" >&2
    exit 1
fi
cd -- "$train_dir"
train_dir="$PWD"

for required in train_all.py config.yaml config_cipt_b5b_entropy_rd.yaml domainbed/lib/cipt_lr_schedule.py; do
    [[ -f "$required" ]] || { printf 'ERROR: missing %s/%s\n' "$train_dir" "$required" >&2; exit 1; }
done
current_branch="$(git branch --show-current)"
[[ "$current_branch" == "$expected_branch" ]] || {
    printf 'ERROR: expected branch %s; current branch is %s.\n' "$expected_branch" "$current_branch" >&2
    exit 1
}

python_name="${CIPT_PYTHON:-python}"
python_bin="$(command -v -- "$python_name")" || { printf 'ERROR: Python not found: %s\n' "$python_name" >&2; exit 1; }
python_bin="$(cd -- "$(dirname -- "$python_bin")" && pwd -P)/$(basename -- "$python_bin")"
data_dir="${CIPT_DATA_DIR:-/home/hooasia/lgg/data/repro_dccl_data}"
clip_path="${CIPT_CLIP_PATH:-/home/hooasia/.cache/clip/ViT-B-16.pt}"
output_parent="${CIPT_OUTPUT_PARENT:-$train_dir/train_output/ablation/gpu0_center_mse_lr2e3_2exp}"
[[ "$output_parent" == /* ]] || output_parent="$train_dir/$output_parent"
session_name="${CIPT_TMUX_SESSION:-cipt_gpu0_center_mse_lr2e3_2exp}"

if [[ "$mode" != --dry-run ]]; then
    [[ -d "$data_dir" ]] || { printf 'ERROR: dataset directory not found: %s\n' "$data_dir" >&2; exit 1; }
    [[ -f "$clip_path" ]] || { printf 'ERROR: CLIP checkpoint not found: %s\n' "$clip_path" >&2; exit 1; }
    command -v flock >/dev/null || { echo 'ERROR: flock is required.' >&2; exit 1; }
    command -v nvidia-smi >/dev/null || { echo 'ERROR: nvidia-smi is required.' >&2; exit 1; }
    nvidia-smi -i 0 --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
fi

if [[ "$mode" == --tmux ]]; then
    command -v tmux >/dev/null || { echo 'ERROR: tmux is required for --tmux.' >&2; exit 1; }
    if tmux has-session -t "=$session_name" 2>/dev/null; then
        printf 'ERROR: tmux session exists. View it with: tmux attach -t %s\n' "$session_name" >&2
        exit 1
    fi
    printf -v launch_command '%q ' env \
        "PATH=$PATH" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" \
        "CIPT_PYTHON=$python_bin" "CIPT_TRAIN_DIR=$train_dir" \
        "CIPT_DATA_DIR=$data_dir" "CIPT_CLIP_PATH=$clip_path" \
        "CIPT_OUTPUT_PARENT=$output_parent" "CIPT_TMUX_SESSION=$session_name" \
        bash "$script_path" --run
    tmux new-session -d -s "$session_name" -c "$train_dir"
    tmux send-keys -t "$session_name" -l "$launch_command"
    tmux send-keys -t "$session_name" Enter
    printf 'Started GPU0 queue. View it with: tmux attach -t %s\n' "$session_name"
    exit 0
fi

if [[ "$mode" == --dry-run ]]; then
    run_root="$output_parent/DRY_RUN"
else
    mkdir -p -- "$output_parent"
    exec 9>"$output_parent/gpu0_center_mse_lr2e3_2exp.lock"
    flock -n 9 || { echo 'ERROR: this queue is already running.' >&2; exit 1; }
    run_root="$(mktemp -d -- "$output_parent/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
    cp -- "$script_path" "$run_root/launcher_snapshot.sh"
    cp -- config.yaml "$run_root/base_config_snapshot.yaml"
    cp -- config_cipt_b5b_entropy_rd.yaml "$run_root/config_snapshot.yaml"
    git rev-parse HEAD > "$run_root/git_commit.txt"
    printf '%s\n' "$current_branch" > "$run_root/git_branch.txt"
    git status --short > "$run_root/git_status.txt"
    printf 'id\tname\tlr\tschedule\tsteps\tstatus\ttime\toutput_root\n' > "$run_root/status.tsv"
    printf '#!/usr/bin/env bash\nset -Eeuo pipefail\ncd -- %q\n' "$train_dir" > "$run_root/commands.sh"
fi

export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
printf 'TRAIN_DIR=%s\nPYTHON=%s\nOUTPUT_ROOT=%s\n' "$train_dir" "$python_bin" "$run_root"
printf 'GPU=0; PACS targets=0,1,2,3; E=S=1; IND=0; center_mse=1; all80; LR=2e-3 constant; contrastive OFF then NR=1 (tau=0.1, warmup=0).\n'

common_args=(
    config_cipt_b5b_entropy_rd.yaml
    --dataset PACS --algorithm CIPTDCCL --test_envs 0 1 2 3
    --data_dir "$data_dir" --cipt_clip_path "$clip_path"
    --deterministic --trial_seed 0 --seed 0 --aug 0
    --checkpoint_freq 100 --batch_size 32 --holdout_fraction 0.2
    --num_workers 2 --eval_num_workers 0
    --cipt_objective env_anchor
    --cipt_template_mode b5b --cipt_selector_mode random
    --cipt_env_selection all --cipt_env_topk 80
    --cipt_env_constraint center_mse --cipt_env_loss_weight 1.0
    --cipt_rd_anchor_mode full
    --cipt_use_de true --cipt_beta 1.0
    --cipt_e_cls_weight 1.0 --cipt_s_uniform_weight 1.0
    --cipt_use_ind false --cipt_gamma 0.0
    --cipt_use_tda false --cipt_env_inference e_only
    --cipt_env_kl_weight 0.0 --cipt_fused_cls_weight 0.0
    --cipt_pure false --cipt_use_aug_view false
    --cipt_class_text_mode learnable
    --cipt_prompt_length 16 --cipt_prompt_init "a photo of a"
    --optimizer adam --swad LossValley
)

run_one() {
    local id="$1" name="$2" subdir="$3" initial_lr="$4" schedule="$5" steps="$6" contrastive="$7"
    local exp_root="$run_root/$subdir"
    local -a command=(
        "$python_bin" -u train_all.py "$name" "${common_args[@]}"
        --output_root "$exp_root" --steps "$steps"
        --lr "$initial_lr" --cipt_lr_schedule "$schedule"
    )
    if [[ "$contrastive" == true ]]; then
        command+=(
            --cipt_use_contrastive true --cipt_contrastive_type neighbor_retention
            --cipt_causal_contrastive_weight 1.0
            --cipt_contrastive_temperature 0.1 --cipt_contrastive_warmup_steps 0
            --cipt_neighbor_k 10 --cipt_neighbor_alpha 4.0
        )
    else
        command+=(--cipt_use_contrastive false --cipt_causal_contrastive_weight 0.0)
    fi

    if [[ "$mode" == --dry-run ]]; then
        printf '\n# Experiment %s\nCUDA_VISIBLE_DEVICES=0 ' "$id"
        printf '%q ' "${command[@]}"
        printf '\n'
        return 0
    fi

    mkdir -- "$exp_root"
    {
        printf '#!/usr/bin/env bash\nset -Eeuo pipefail\ncd -- %q\nCUDA_VISIBLE_DEVICES=0 ' "$train_dir"
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "$exp_root/command.sh"
    {
        printf '\nCUDA_VISIBLE_DEVICES=0 '
        printf '%q ' "${command[@]}"
        printf '\n'
    } >> "$run_root/commands.sh"
    printf '\n[%s] START [%s/2] %s\n' "$(date -Is)" "$id" "$name" | tee -a "$run_root/serial.log"
    printf '%s\t%s\t%s\t%s\t%s\tSTART\t%s\t%s\n' \
        "$id" "$name" "$initial_lr" "$schedule" "$steps" "$(date -Is)" "$exp_root" >> "$run_root/status.tsv"

    if "${command[@]}" 2>&1 | tee "$exp_root/console.log"; then
        printf '%s\t%s\t%s\t%s\t%s\tSUCCESS\t%s\t%s\n' \
            "$id" "$name" "$initial_lr" "$schedule" "$steps" "$(date -Is)" "$exp_root" >> "$run_root/status.tsv"
        printf '[%s] SUCCESS [%s/2] %s\n' "$(date -Is)" "$id" "$name" | tee -a "$run_root/serial.log"
    else
        local exit_code=$?
        printf '%s\t%s\t%s\t%s\t%s\tFAILED(%s)\t%s\t%s\n' \
            "$id" "$name" "$initial_lr" "$schedule" "$steps" "$exit_code" "$(date -Is)" "$exp_root" >> "$run_root/status.tsv"
        printf '[%s] FAILED [%s/2], exit=%s. Queue stopped. Log: %s/console.log\n' \
            "$(date -Is)" "$id" "$exit_code" "$exp_root" | tee -a "$run_root/serial.log" >&2
        exit "$exit_code"
    fi
}

run_one 01 pacs_es1_ind0_center_mse1_all80_nocon_lr2e3_constant_3000 \
    01_center_mse_nocon_lr2e3_constant_3000 2e-3 constant 3000 false
run_one 02 pacs_es1_ind0_center_mse1_all80_nr1_lr2e3_constant_3000 \
    02_center_mse_nr1_lr2e3_constant_3000 2e-3 constant 3000 true

if [[ "$mode" != --dry-run ]]; then
    printf '\n[%s] BOTH EXPERIMENTS COMPLETED. Results: %s\n' "$(date -Is)" "$run_root" | tee -a "$run_root/serial.log"
    cat -- "$run_root/status.tsv"
fi
