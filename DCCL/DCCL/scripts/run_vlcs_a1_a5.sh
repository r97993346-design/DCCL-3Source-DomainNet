#!/usr/bin/env bash
# Five DICM controls on VLCS; A0 is not included. --show reviews without training.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON:-python}"
RUN_SEED="${SEED:-0}"
SPLIT_SEED="${TRIAL_SEED:-0}"
RUN_STEPS="${STEPS:-5000}"
EVAL_FREQ="${CHECKPOINT_FREQ:-100}"
RUN_OUTPUT_ROOT="${OUTPUT_ROOT:-train_output}"

for mode in A1 A2 A3 A4 A5; do
  "$PYTHON_BIN" -u train_all.py "vlcs_${mode}_seed${RUN_SEED}_n${RUN_STEPS}" configs/dicm_vlcs.yaml \
    --algorithm DICM --dataset VLCS --model resnet50 --dicm_ablation "$mode" \
    --seed "$RUN_SEED" --trial_seed "$SPLIT_SEED" --deterministic \
    --steps "$RUN_STEPS" --checkpoint_freq "$EVAL_FREQ" --num_workers "${NUM_WORKERS:-4}" \
    --output_root "$RUN_OUTPUT_ROOT" "$@"
done
