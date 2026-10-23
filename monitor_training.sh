#!/bin/bash

LOG="training_resource_monitor.log"
INTERVAL=20

echo "===== Monitor started at $(date) =====" >> "$LOG"

while true; do
    echo "" >> "$LOG"
    echo "==================================================" >> "$LOG"
    echo "$(date)" >> "$LOG"
    echo "==================================================" >> "$LOG"

    echo "[SYSTEM MEMORY]" >> "$LOG"
    free -h >> "$LOG"

    echo "" >> "$LOG"
    echo "[GPU]" >> "$LOG"
    nvidia-smi \
      --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader >> "$LOG"

    echo "" >> "$LOG"
    echo "[TRAINING PROCESSES]" >> "$LOG"

    ps -eo pid,ppid,rss,%mem,%cpu,etime,cmd --sort=-rss | \
    grep -E "train_all.py|python.*domainbed|pt_data_worker" | \
    grep -v grep >> "$LOG"

    echo "" >> "$LOG"
    echo "[DATALOADER SUMMARY]" >> "$LOG"

    ps -eo rss,cmd | \
    awk '
    /pt_data_worker/ {
        sum += $1;
        n++
    }
    END {
        printf "worker_count=%d  total_worker_RSS=%.2f GB\n",
        n, sum/1024/1024
    }' >> "$LOG"

    echo "" >> "$LOG"
    echo "[TRAIN PROCESS SUMMARY]" >> "$LOG"

    ps -eo pid,rss,%mem,%cpu,etime,cmd | \
    awk '
    /train_all.py/ && !/awk/ {
        printf "PID=%s RSS=%.2fGB MEM=%s%% CPU=%s%% ELAPSED=%s CMD=%s\n",
        $1,$2/1024/1024,$3,$4,$5,$6
    }' >> "$LOG"

    AVAILABLE_MB=$(free -m | awk '/Mem:/ {print $7}')

    if [ "$AVAILABLE_MB" -lt 8192 ]; then
        echo "!!! WARNING: available RAM below 8 GB !!!" >> "$LOG"
    fi

    if [ "$AVAILABLE_MB" -lt 4096 ]; then
        echo "!!! CRITICAL: available RAM below 4 GB !!!" >> "$LOG"
    fi

    sleep "$INTERVAL"
done
