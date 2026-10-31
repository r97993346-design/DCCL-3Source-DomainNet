#!/usr/bin/env bash
# PACS: class-agnostic B5a + adaptive TDA + neighborhood-retention contrastive.
# Run all four experiments sequentially on physical GPU 1. No training is
# launched until this file is explicitly run. --dry-run only prints commands.
# Optional path/interpreter overrides: CIPT_REPO_DIR, CIPT_DATA_DIR,
# CIPT_CLIP_PATH, CIPT_PYTHON, CIPT_OUTPUT_ROOT.
set -Eeuo pipefail

DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  -h|--help)
    printf 'Usage: bash %s [--dry-run]\n' "${BASH_SOURCE[0]}"
    exit 0
    ;;
  *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  printf 'Only --dry-run or --help is accepted.\n' >&2
  exit 2
fi

REPO_DIR="${CIPT_REPO_DIR:-/home/hooasia/lgg/projects/cipt_gpu0}"
DATA_DIR="${CIPT_DATA_DIR:-/home/hooasia/lgg/data/repro_dccl_data}"
CLIP_PATH="${CIPT_CLIP_PATH:-/home/hooasia/.cache/clip/ViT-B-16.pt}"
PYTHON_BIN="${CIPT_PYTHON:-python}"
EXPECTED_BRANCH='feature/cipt-b5b-entropy-mcc-rd'
SCRIPT_PATH="$(realpath -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

cd -- "$REPO_DIR"
ACTUAL_BRANCH="$(git branch --show-current)"
if [[ "$ACTUAL_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  printf 'Expected branch %s; found %s. No training started.\n' \
    "$EXPECTED_BRANCH" "$ACTUAL_BRANCH" >&2
  exit 1
fi
REPO_DIR="$(pwd -P)"
WORK_DIR="$REPO_DIR/DCCL/DCCL"
cd -- "$WORK_DIR"
for required_file in train_all.py config.yaml config_cipt_original_visualization.yaml \
    domainbed/lib/cipt_lr_schedule.py; do
  if [[ ! -f "$required_file" ]]; then
    printf 'Missing required file: %s/%s\n' "$WORK_DIR" "$required_file" >&2
    exit 1
  fi
done

OUTPUT_PARENT="${CIPT_OUTPUT_ROOT:-$WORK_DIR/train_output/ablation/b5a_adaptive_nr_gpu1}"
if [[ "$OUTPUT_PARENT" != /* ]]; then
  OUTPUT_PARENT="$WORK_DIR/$OUTPUT_PARENT"
fi

if (( DRY_RUN )); then
  RUN_DIR="$OUTPUT_PARENT/DRY_RUN"
else
  command -v "$PYTHON_BIN" >/dev/null
  command -v flock >/dev/null
  if [[ ! -d "$DATA_DIR" || ! -f "$CLIP_PATH" ]]; then
    printf 'Check DATA_DIR=%s and CLIP_PATH=%s. No training started.\n' \
      "$DATA_DIR" "$CLIP_PATH" >&2
    exit 1
  fi
  # Verify legacy cosine compatibility on CPU without loading CLIP or images.
  "$PYTHON_BIN" - <<'PY'
from types import SimpleNamespace
import torch
from domainbed.lib.cipt_lr_schedule import make_cipt_lr_scheduler

parameter = torch.nn.Parameter(torch.tensor(1.0))
algorithm = SimpleNamespace(optimizer=torch.optim.Adam([parameter], lr=0.002))
scheduler = make_cipt_lr_scheduler(
    algorithm,
    {"cipt_objective": "legacy", "cipt_lr_schedule": "cosine"},
    3001,
)
assert scheduler is not None, "legacy cosine scheduler was not created"
PY
  mkdir -p -- "$OUTPUT_PARENT"
  exec 9>"$OUTPUT_PARENT/.gpu1_serial.lock"
  if ! flock -n 9; then
    printf 'This GPU1 experiment queue is already running.\n' >&2
    exit 1
  fi
  RUN_DIR="$(mktemp -d "$OUTPUT_PARENT/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
  exec > >(tee -a "$RUN_DIR/serial.log") 2>&1
  printf 'time\texperiment\tstatus\texit_code\n' > "$RUN_DIR/status.tsv"
  git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
  git status --short > "$RUN_DIR/git_status.txt"
  cp -- config.yaml config_cipt_original_visualization.yaml "$RUN_DIR/"
  cp -- "$SCRIPT_PATH" "$RUN_DIR/launcher_snapshot.sh"
fi

COMMON_ARGS=(
  config_cipt_original_visualization.yaml
  --dataset PACS --algorithm CIPTDCCL
  --data_dir "$DATA_DIR" --cipt_clip_path "$CLIP_PATH"
  --test_envs 0 1 2 3
  --deterministic --trial_seed 0 --seed 0
  --checkpoint_freq 100 --aug 0
  --batch_size 32 --optimizer adam --weight_decay 0
  --cipt_clip_backbone ViT-B/16
  --cipt_objective legacy
  --cipt_template_mode b5a --cipt_neutral_subject subject
  --cipt_beta 4 --cipt_gamma 5
  --cipt_prompt_length 16 --cipt_prompt_init 'a photo of a'
  --cipt_tda_heads 1
  --cipt_selector_mode adaptive --cipt_selector_candidates 8 --cipt_k 4
  --cipt_use_de true --cipt_use_ind true --cipt_use_tda true
  --cipt_use_aug_view false
  --cipt_pure false --cipt_use_contrastive true
  --cipt_contrastive_type neighbor_retention
  --cipt_causal_contrastive_weight 1
  --cipt_contrastive_warmup_steps 0 --cipt_contrastive_temperature 0.1
  --cipt_neighbor_k 5 --cipt_neighbor_alpha 0.5
  --cipt_env_kl_weight 0 --cipt_fused_cls_weight 0
  --cipt_lr_min 0
  --swad LossValley
)

print_command() {
  printf 'CUDA_VISIBLE_DEVICES=1 '
  printf '%q ' "$@"
  printf '\n'
}

run_one() {
  local name="$1" lr="$2" schedule="$3" steps="$4"
  local exp_dir="$RUN_DIR/$name"
  local rc
  local -a cmd=(
    "$PYTHON_BIN" -u train_all.py "$name"
    "${COMMON_ARGS[@]}"
    --output_root "$exp_dir/train_output"
    --lr "$lr" --cipt_lr_schedule "$schedule" --steps "$steps"
  )
  if (( DRY_RUN )); then
    print_command "${cmd[@]}"
    return
  fi

  mkdir -p -- "$exp_dir"
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'cd -- %q\n' "$WORK_DIR"
    print_command "${cmd[@]}"
  } > "$exp_dir/command.sh"
  printf '\n[%s] START %s (GPU=1, lr=%s, schedule=%s, steps=%s)\n' \
    "$(date --iso-8601=seconds)" "$name" "$lr" "$schedule" "$steps"
  printf '%s\t%s\tRUNNING\t-\n' "$(date --iso-8601=seconds)" "$name" \
    >> "$RUN_DIR/status.tsv"
  print_command "${cmd[@]}"

  if "${cmd[@]}" 2>&1 | tee "$exp_dir/console.log"; then
    printf '%s\t%s\tOK\t0\n' "$(date --iso-8601=seconds)" "$name" \
      >> "$RUN_DIR/status.tsv"
    printf '[%s] DONE %s\n' "$(date --iso-8601=seconds)" "$name"
  else
    rc=$?
    printf '%s\t%s\tFAILED\t%s\n' "$(date --iso-8601=seconds)" "$name" "$rc" \
      >> "$RUN_DIR/status.tsv"
    printf 'Experiment failed (exit %s): %s. Queue stopped; see %s/console.log\n' \
      "$rc" "$name" "$exp_dir" >&2
    exit "$rc"
  fi
}

printf 'GPU1 sequential PACS queue: b5a/subject, adaptive 8->4, beta4/gamma5, NR1, warmup0, temp0.1\n'
if (( DRY_RUN )); then
  printf 'DRY RUN: commands only; no training or output folders are created.\n'
else
  printf 'Results and logs: %s\n' "$RUN_DIR"
fi

run_one 01_pacs_b5a_adaptive_nr1_lr002_cos_n3000   0.002 cosine   3000
run_one 02_pacs_b5a_adaptive_nr1_lr1e5_cos_n3000   1e-5  cosine   3000
run_one 03_pacs_b5a_adaptive_nr1_lr1e5_fixed_n3000 1e-5  constant 3000
run_one 04_pacs_b5a_adaptive_nr1_lr002_cos_n5000   0.002 cosine   5000

if (( ! DRY_RUN )); then
  printf '[%s] ALL FOUR EXPERIMENTS COMPLETED: %s\n' \
    "$(date --iso-8601=seconds)" "$RUN_DIR"
fi
