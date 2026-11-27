#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/code-terra/DCCL/DCCL
scripts=scripts/terra_fixed6
root=/root/autodl-tmp/outputs/terra_fixed6_tpe_lr2e5_prompt_rd3_condcov3_20261127
python=/root/miniconda3/envs/cipt/bin/python
action=${1:-start}
case "$action" in
  prepare|pause|stop|authorize-start) exec "$python" -B "$scripts/run.py" "$action" ;;
  start|resume) ;;
  *) exit 2 ;;
esac
"$python" -B "$scripts/run.py" prepare
"$python" -B "$scripts/run.py" ready
command -v tmux >/dev/null
status=$(mktemp "$root/.status.XXXXXXXX")
printf '{"state":"launching"}\n' > "$status"
mv -- "$status" "$root/status.json"
mode=run
[[ "$action" != resume ]] || mode=resume
tmux new-session -d -s terra-fixed6 -c "$PWD"   "exec '$python' -B -u '$scripts/run.py' '$mode' >>'$root/controller.log' 2>&1"
tmux new-session -d -s terra-fixed6-sync -c "$PWD"   "exec '$python' -B -u '$scripts/sync.py' >>'$root/github_sync.log' 2>&1" || tmux has-session -t terra-fixed6-sync
