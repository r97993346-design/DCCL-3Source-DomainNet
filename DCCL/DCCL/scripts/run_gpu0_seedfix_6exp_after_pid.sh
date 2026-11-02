#!/usr/bin/env bash
# Wait only for the requested PID, then run the approved six PACS jobs on GPU0.
# Usage: bash scripts/run_gpu0_seedfix_6exp_after_pid.sh --wait-pid 3112421
# Preview only: bash scripts/run_gpu0_seedfix_6exp_after_pid.sh --dry-run
set -Eeuo pipefail

WAIT_PID=3112421
DRY_RUN=0
while (($#)); do
  case "$1" in
    --wait-pid) WAIT_PID="${2:?Missing PID}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$WAIT_PID" =~ ^[0-9]+$ ]] || { printf 'Invalid PID\n' >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${CIPT_PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${CIPT_PYTHON_BIN:-/home/hooasia/miniconda3/envs/dccl/bin/python}"
DATA_DIR="${CIPT_DATA_DIR:-/home/hooasia/lgg/data/repro_dccl_data}"
CLIP_PATH="${CIPT_CLIP_PATH:-/home/hooasia/.cache/clip/ViT-B-16.pt}"
OUTPUT_ROOT="${CIPT_OUTPUT_ROOT:-$PROJECT_DIR/train_output/ablation/seedfix_lr5e5_es1_ind_center_mse_20261102}"
POLL_SECONDS="${CIPT_POLL_SECONDS:-10}"
EXPECTED_BRANCH=feature/cipt-per-target-seed-20261102
RUN_ROOT=

log() { printf '[%s] %s\n' "$(TZ=Asia/Shanghai date '+%F %T%z')" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

[[ "$POLL_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ && ! "$POLL_SECONDS" =~ ^0+([.]0+)?$ ]] || die 'Invalid poll interval'
cd -- "$PROJECT_DIR" || die "Cannot enter $PROJECT_DIR"
[[ -f train_all.py && -f config_cipt_b5b_all80_center_mse.yaml ]] || die 'Missing training entry/config'
[[ "$(git branch --show-current)" == "$EXPECTED_BRANCH" ]] || die "Expected branch $EXPECTED_BRANCH"

SPECS=(
  '01_pacs_es1_ind0_n5000|5000|false|0|none|0'
  '02_pacs_es1_ind01_n5000|5000|true|0.1|none|0'
  '03_pacs_es1_ind02_n5000|5000|true|0.2|none|0'
  '04_pacs_es1_ind05_n5000|5000|true|0.5|none|0'
  '05_pacs_es1_ind0_center_mse1_n5000|5000|false|0|center_mse|1'
  '06_pacs_es1_ind0_center_mse1_n3800|3800|false|0|center_mse|1'
)

build_command() {
  local name="$1" steps="$2" use_ind="$3" gamma="$4" constraint="$5" env_weight="$6"
  COMMAND=(
    "$PYTHON_BIN" -u train_all.py "$name" config_cipt_b5b_all80_center_mse.yaml
    --dataset PACS --algorithm CIPTDCCL
    --data_dir "$DATA_DIR" --cipt_clip_path "$CLIP_PATH"
    --output_root "${RUN_ROOT:-$OUTPUT_ROOT/DRY_RUN}/$name"
    --test_envs 0 1 2 3
    --lr 5e-5 --cipt_lr_schedule constant --optimizer adam --weight_decay 0.0
    --batch_size 32 --num_workers 2 --eval_num_workers 2
    --seed 0 --trial_seed 0 --deterministic --checkpoint_freq 100 --aug 0
    --swad LossValley
    --cipt_objective env_anchor --cipt_use_de true
    --cipt_e_cls_weight 1 --cipt_s_uniform_weight 1
    --cipt_use_tda false --cipt_env_inference e_only
    --cipt_env_selection all --cipt_env_topk 80 --cipt_class_text_mode learnable
    --cipt_use_contrastive false --cipt_causal_contrastive_weight 0
    --steps "$steps" --cipt_use_ind "$use_ind" --cipt_gamma "$gamma"
    --cipt_env_constraint "$constraint" --cipt_env_loss_weight "$env_weight"
  )
}

print_command() {
  printf 'CUDA_VISIBLE_DEVICES=0 '
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
}

if ((DRY_RUN)); then
  log "DRY RUN: wait only for PID $WAIT_PID to exit; then run six serial jobs on GPU0."
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r name steps use_ind gamma constraint env_weight <<< "$spec"
    build_command "$name" "$steps" "$use_ind" "$gamma" "$constraint" "$env_weight"
    print_command
  done
  exit 0
fi

[[ -x "$PYTHON_BIN" ]] || die "Missing interpreter $PYTHON_BIN"
[[ -d "$DATA_DIR/PACS" ]] || die "Missing PACS data: $DATA_DIR/PACS"
[[ -f "$CLIP_PATH" ]] || die "Missing CLIP checkpoint: $CLIP_PATH"
command -v flock >/dev/null || die 'flock is unavailable'
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
mkdir -p -- "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.gpu0_queue.lock"
flock -n 9 || die 'Another copy of this GPU0 queue is already running'
RUN_ROOT="$(mktemp -d "$OUTPUT_ROOT/run_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')_XXXXXX")"
printf 'time\texperiment\tstatus\texit_code\n' > "$RUN_ROOT/status.tsv"
log "Queue output: $RUN_ROOT"

status() {
  printf '%s\t%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T%z')" "$1" "$2" "${3:-}" >> "$RUN_ROOT/status.tsv"
}
finish() {
  local code=$?
  status QUEUE EXIT "$code"
  log "Queue exited with code $code; status: $RUN_ROOT/status.tsv"
}
trap finish EXIT

# Use Linux's start-time ticks rather than PID alone, so PID reuse cannot
# accidentally attach this queue to a new process. Zombies count as exited.
process_identity() {
  "$PYTHON_BIN" - "$WAIT_PID" <<'PY'
from pathlib import Path
import sys
process = Path('/proc') / sys.argv[1]
try:
    fields = (process / 'stat').read_text().rsplit(')', 1)[1].split()
except (FileNotFoundError, ProcessLookupError):
    sys.exit(0)
if fields[0] != 'Z':
    print(fields[19])
PY
}

if [[ "$WAIT_PID" != 0 ]]; then
  original_identity="$(process_identity)"
  if [[ -n "$original_identity" ]]; then
    log "Waiting for original training PID $WAIT_PID to exit."
    status QUEUE "WAIT_PID_$WAIT_PID"
    while [[ "$(process_identity)" == "$original_identity" ]]; do
      sleep "$POLL_SECONDS"
    done
    log "Original training PID $WAIT_PID has exited."
  else
    log "PID $WAIT_PID has already exited; starting the queue."
  fi
fi

for spec in "${SPECS[@]}"; do
  IFS='|' read -r name steps use_ind gamma constraint env_weight <<< "$spec"
  build_command "$name" "$steps" "$use_ind" "$gamma" "$constraint" "$env_weight"
  mkdir -p -- "$RUN_ROOT/$name"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'cd -- %q\n' "$PROJECT_DIR"
    print_command
  } > "$RUN_ROOT/$name/command.sh"
  status "$name" START
  log "Starting $name: fixed LR 5e-5, steps=$steps, IND=$gamma, Center-MSE=$env_weight."
  if CUDA_VISIBLE_DEVICES=0 "${COMMAND[@]}" 2>&1 | tee "$RUN_ROOT/$name/console.log"; then
    status "$name" DONE 0
    log "Completed $name."
  else
    exit_codes=("${PIPESTATUS[@]}")
    code="${exit_codes[0]}"
    if ((code == 0)); then code="${exit_codes[1]:-1}"; fi
    status "$name" FAILED "$code"
    die "Experiment $name failed (exit $code); remaining experiments will not start."
  fi
done
log 'All six experiments completed.'
