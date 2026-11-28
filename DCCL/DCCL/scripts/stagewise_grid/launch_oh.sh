#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
base=/root/autodl-tmp
out="$base/outputs/oh_stagewise_grid_crossdomain_constant1e5_20261128"
py=/root/miniconda3/envs/cipt/bin/python
test -d "$base/vlcs-grid-code/DCCL/DCCL"
test -f "$base/weights/ViT-B-16.pt"
if ! tmux has-session -t '=oh-stagewise-grid' 2>/dev/null; then
    mkdir -p "$out"
    "$py" -B scripts/stagewise_grid/verify_officehome.py | tee "$out/dataset_verification.log"
    tmux new-session -d -s oh-stagewise-grid -c "$PWD" "exec '$py' -B -u scripts/stagewise_grid/run.py >>'$out/controller.log' 2>&1"
fi
if ! tmux has-session -t '=oh-stagewise-grid-sync' 2>/dev/null; then
    tmux new-session -d -s oh-stagewise-grid-sync -c "$PWD" "exec '$py' -B -u scripts/stagewise_grid/sync.py >>'$out/github_sync.log' 2>&1"
fi
tmux list-panes -a -F '#{session_name} #{pane_pid}' | grep '^oh-stagewise-grid'
