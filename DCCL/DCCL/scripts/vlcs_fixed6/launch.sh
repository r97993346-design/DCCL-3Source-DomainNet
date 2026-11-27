#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/code-vlcs/DCCL/DCCL
scripts=scripts/vlcs_fixed6
root=/root/autodl-tmp/outputs/vlcs_fixed6_tpe_lr1e5_prompt_rd3_condcov3_20261127
python=/root/miniconda3/envs/cipt/bin/python
action=${1:-start}
case "$action" in
  prepare|pause|stop) exec "$python" -B "$scripts/run.py" "$action" ;;
  start|resume) ;;
  *) exit 2 ;;
esac
"$python" -B "$scripts/run.py" prepare
"$python" -B "$scripts/run.py" ready
status=$(mktemp "$root/.status.XXXXXXXX")
printf '{"state":"launching"}\n' > "$status"
mv -- "$status" "$root/status.json"
mode=run
[[ "$action" != resume ]] || mode=resume
nohup setsid "$python" -B -u "$scripts/run.py" "$mode" </dev/null >>"$root/controller.log" 2>&1 &
echo "Controller PID $!"
nohup setsid "$python" -B -u "$scripts/sync.py" </dev/null >>"$root/github_sync.log" 2>&1 &
