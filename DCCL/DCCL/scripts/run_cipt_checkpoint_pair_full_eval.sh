#!/usr/bin/env bash
# Evaluate exact checkpoints 2 and 3 on the same full splits, then upload results.
set -Eeuo pipefail
CIPT_PROJECT_DIR="${CIPT_PROJECT_DIR:-/home/hooasia/lgg/projects/repro_dccl_3domain/DCCL/DCCL}"
CIPT_DATA_DIR="${CIPT_DATA_DIR:-/home/hooasia/lgg/data/repro_dccl_data}"
CIPT_CLIP_PATH="${CIPT_CLIP_PATH:-/home/hooasia/.cache/clip/ViT-B-16.pt}"
CIPT_PYTHON="${CIPT_PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CIPT_GPU:-1}"
export PYTHONUNBUFFERED=1
CIPT_EVAL_TMP=$(mktemp -d /tmp/cipt-pair-eval.XXXXXX)
CIPT_EVAL_TREE="$CIPT_EVAL_TMP/repository"
CIPT_EVAL_BRANCH="analysis/cipt-pair-full-eval-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-${CIPT_EVAL_TMP##*.}"
CIPT_EVAL_REL="DCCL/DCCL/analysis_results/cipt_checkpoints_2_3_full_eval"
CIPT_EVAL_BASE=1e97e3d31077711015c6a2ebd0874a7d8f05652e
CIPT_CHECKPOINT_2="$CIPT_PROJECT_DIR/train_output/cipt_mechanism/PACS/261023_14-43-14_cipt_original_te0_seed0/checkpoints/TE0_5000.pth"
CIPT_CHECKPOINT_3="$CIPT_PROJECT_DIR/train_output/cipt_mechanism/PACS/261025_23-42-59_cipt_subject80_te0_seed0/checkpoints/TE0_5000.pth"
for CIPT_REQUIRED in "$CIPT_CHECKPOINT_2" "$CIPT_CHECKPOINT_3" "$CIPT_CLIP_PATH"; do
    if [[ ! -f "$CIPT_REQUIRED" ]]; then
        printf 'Missing required file: %s\n' "$CIPT_REQUIRED" >&2
        exit 1
    fi
done
"$CIPT_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable in this Python environment.')
print('Evaluation GPU:', torch.cuda.get_device_name(0), flush=True)
PY
git -C "$CIPT_PROJECT_DIR" worktree add -b "$CIPT_EVAL_BRANCH" "$CIPT_EVAL_TREE" "$CIPT_EVAL_BASE"
CIPT_EVAL_CODE="$CIPT_EVAL_TREE/DCCL/DCCL"
CIPT_EVAL_OUT="$CIPT_EVAL_TREE/$CIPT_EVAL_REL"
export CIPT_EVAL_OUT CIPT_EVAL_BASE
mkdir -p "$CIPT_EVAL_OUT/logs"
for CIPT_CASE in 2 3; do
    if [[ "$CIPT_CASE" == 2 ]]; then
        CIPT_LABEL=02_original
        CIPT_CHECKPOINT="$CIPT_CHECKPOINT_2"
    else
        CIPT_LABEL=03_subject80
        CIPT_CHECKPOINT="$CIPT_CHECKPOINT_3"
    fi
    "$CIPT_PYTHON" -u "$CIPT_EVAL_CODE/scripts/audit_cipt_checkpoint_pair.py" \
        --repo-root "$CIPT_EVAL_CODE" \
        --checkpoint "$CIPT_CHECKPOINT" \
        --data-dir "$CIPT_DATA_DIR" --clip-path "$CIPT_CLIP_PATH" \
        --device cuda --batch-size 64 --num-workers 0 \
        --output-dir "$CIPT_EVAL_OUT/$CIPT_LABEL" \
        2>&1 | tee "$CIPT_EVAL_OUT/logs/$CIPT_LABEL.log"
done
"$CIPT_PYTHON" - <<'PY'
import csv
import json
import os
from pathlib import Path
root = Path(os.environ['CIPT_EVAL_OUT'])
reports = {label: json.loads((root / label / 'report.json').read_text())
           for label in ('02_original', '03_subject80')}
a, b = reports.values()
for key in ('classes', 'test_envs', 'partitions', 'template_indices'):
    if a[key] != b[key]:
        raise ValueError(f'The two evaluations do not match on {key}')
expected = {
    '02_original': 'fa03ca305faf18feba0347058713910a3c8664bf750b5e6c9cc2e3f58b860a42',
    '03_subject80': '6c02a511830eab9245868b83662efd091cb0190e821e04b5b70e7e0b57e17e25',
}
rows = []
for label, report in reports.items():
    if report['checkpoint_sha256'] != expected[label]:
        raise ValueError(f'Wrong checkpoint: {label}')
    if report['coverage'] != 'FULL_DATASET_SPLITS':
        raise ValueError(f'Partial evaluation cannot be uploaded as full: {label}')
    for split, modes in report['metrics'].items():
        for mode, values in modes.items():
            pooled = values['pooled']
            rows.append({'model': label, 'split': split, 'mode': mode,
                         'n': pooled['n'], 'accuracy_pct': 100 * pooled['accuracy'],
                         'domain_macro_accuracy_pct': 100 * values['domain_macro_accuracy'],
                         'mean_CE': pooled['CE_of_mean_logits'],
                         'largest_predicted_class': pooled['largest_predicted_class'],
                         'largest_predicted_fraction': pooled['largest_predicted_fraction']})
            if mode in ('native_TDA', 'E_only'):
                print(f"{label:14s} {split:12s} {mode:10s} pooled={100*pooled['accuracy']:.3f}% "
                      f"macro={100*values['domain_macro_accuracy']:.3f}%", flush=True)
with (root / 'comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root / 'run_manifest.json').write_text(json.dumps({
    'evaluation_code_commit': os.environ['CIPT_EVAL_BASE'],
    'checkpoint_sha256': expected,
    'protocol': 'Full source_train/source_val/target_in/target_out; native TDA, E-only, LN(E), fixed diagnostic alpha values.',
    'same_partitions_and_class_order_verified': True,
    'selection': 'None. Fixed existing weights; target data do not choose a model or alpha.',
}, indent=2))
PY
git -C "$CIPT_EVAL_TREE" add -f -- "$CIPT_EVAL_REL"
git -C "$CIPT_EVAL_TREE" commit --only -m "Upload full-split evaluation of original and subject80 CIPT checkpoints" -- "$CIPT_EVAL_REL"
git -C "$CIPT_EVAL_TREE" push origin "HEAD:refs/heads/$CIPT_EVAL_BRANCH"
printf '\nFull evaluation uploaded.\nGitHub branch: %s\nDirectory: %s\n' "$CIPT_EVAL_BRANCH" "$CIPT_EVAL_REL"
