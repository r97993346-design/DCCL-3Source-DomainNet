#!/usr/bin/env bash
# PACS -> VLCS -> OfficeHome, physical GPU1; source-only IID + dense SWAD.
set -Eeuo pipefail
trap 'printf "\nInterrupted; stopping the serial launcher.\n" >&2; exit 130' INT
trap 'printf "\nTerminated; stopping the serial launcher.\n" >&2; exit 143' TERM

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR"

CIPT_PYTHON="${CIPT_PYTHON:-python}"
CIPT_DATA_DIR="${CIPT_DATA_DIR:-/home/hooasia/lgg/data/repro_dccl_data}"
CIPT_CLIP_PATH="${CIPT_CLIP_PATH:-/home/hooasia/.cache/clip/ViT-B-16.pt}"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

OUTPUT_PARENT=train_output/ablation/cipt_tda_iid_swad_gpu1
mkdir -p "$OUTPUT_PARENT"
RUN_ROOT=$(mktemp -d "$OUTPUT_PARENT/run_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_XXXXXX")
RUN_ROOT=$(cd "$RUN_ROOT" && pwd)
mkdir -p "$RUN_ROOT/logs"
printf '%s\n' "$RUN_ROOT" > "$OUTPUT_PARENT/latest_run.txt"
cp -- "$SCRIPT_PATH" "$RUN_ROOT/launcher_snapshot.sh"
git rev-parse HEAD > "$RUN_ROOT/git_commit.txt" 2>/dev/null || true
exec > >(tee -a "$RUN_ROOT/launcher.log") 2>&1

printf 'GPU1 serial run: PACS -> VLCS -> OfficeHome\nResults: %s\n' "$RUN_ROOT"
if [[ ! -d "$CIPT_DATA_DIR" || ! -f "$CIPT_CLIP_PATH" ]]; then
    printf 'Missing data directory or CLIP checkpoint: %s ; %s\n' "$CIPT_DATA_DIR" "$CIPT_CLIP_PATH" >&2
    exit 1
fi
if [[ ! -f third_party/CIPT/cipt/model.py ]]; then
    printf 'Missing CIPT submodule; run git submodule update --init --recursive.\n' >&2
    exit 1
fi
"$CIPT_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Physical GPU1 is not available to PyTorch; check the active Python environment.")
print("CUDA_VISIBLE_DEVICES=1; logical cuda:0 = physical GPU1:", torch.cuda.get_device_name(0), flush=True)
PY

printf 'dataset\tstarted\tfinished\texit_code\n' > "$RUN_ROOT/status.tsv"
FAILURES=0
for DATASET in PACS VLCS OfficeHome; do
    STARTED=$(TZ=Asia/Shanghai date --iso-8601=seconds)
    COMMAND=(
        "$CIPT_PYTHON" -u train_cipt_tda_iid_swad_domainbed.py
        --dataset "$DATASET"
        --data-dir "$CIPT_DATA_DIR"
        --clip-path "$CIPT_CLIP_PATH"
        --test-envs 0 1 2 3
        --seed 0 --trial-seed 0 --holdout-fraction 0.2
        --epochs 30 --batch-size 64 --eval-batch-size 128 --lr 0.0025
        --beta 4 --gamma 5 --k 4 --n-ctx 16 --num-heads 8
        --model-variant consistent
        --num-workers 4 --eval-num-workers 0
        --swad-n-converge 3 --swad-n-tolerance 6 --swad-tolerance-ratio 0.3
        --eval-every 1 --checkpoint-freq 0
        --output-root "$RUN_ROOT"
    )
    {
        printf 'CUDA_VISIBLE_DEVICES=1 '
        printf '%q ' "${COMMAND[@]}"
        printf '\n'
    } > "$RUN_ROOT/logs/$DATASET.command.sh"
    printf '\n[%s] START %s\n' "$STARTED" "$DATASET"
    if "${COMMAND[@]}" 2>&1 | tee "$RUN_ROOT/logs/$DATASET.log"; then
        EXIT_CODE=0
    else
        EXIT_CODE=$?
        FAILURES=$((FAILURES + 1))
    fi
    FINISHED=$(TZ=Asia/Shanghai date --iso-8601=seconds)
    printf '%s\t%s\t%s\t%s\n' "$DATASET" "$STARTED" "$FINISHED" "$EXIT_CODE" >> "$RUN_ROOT/status.tsv"
    printf '[%s] END %s exit=%s\n' "$FINISHED" "$DATASET" "$EXIT_CODE"
done

printf '\nFinished all three datasets; failures=%s\nStatus: %s/status.tsv\n' "$FAILURES" "$RUN_ROOT"
if (( FAILURES > 0 )); then
    exit 1
fi
