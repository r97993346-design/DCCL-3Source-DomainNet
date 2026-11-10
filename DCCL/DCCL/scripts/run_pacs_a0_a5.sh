#!/usr/bin/env bash
# Prepare/launch matched A0-A5 jobs. Pass --show to review without training.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON:-python}"
RUN_SEED="${SEED:-0}"
SPLIT_SEED="${TRIAL_SEED:-0}"
RUN_STEPS="${STEPS:-5000}"
EVAL_FREQ="${CHECKPOINT_FREQ:-100}"

for mode in A0 A1 A2 A3 A4 A5; do
  if [[ "$mode" == A0 ]]; then
    algorithm=DCCL
    config=configs/dccl_pacs_a0.yaml
    options=()
  else
    algorithm=DICM
    config=configs/dicm_pacs.yaml
    options=(--dicm_ablation "$mode")
  fi
  "$PYTHON_BIN" -u train_all.py "pacs_${mode}_seed${RUN_SEED}_n${RUN_STEPS}" "$config" \
    --algorithm "$algorithm" --dataset PACS --model resnet50 \
    --seed "$RUN_SEED" --trial_seed "$SPLIT_SEED" --deterministic \
    --steps "$RUN_STEPS" --checkpoint_freq "$EVAL_FREQ" "${options[@]}" "$@"
done
