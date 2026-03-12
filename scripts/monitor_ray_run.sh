#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hieule/research/reflector-position"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/results/ray_parallel_monitor"
RUN_LOG="$LOG_DIR/run_with_timeout.log"
GPU_LOG="$LOG_DIR/gpu_mem_samples.csv"

mkdir -p "$LOG_DIR"
: > "$RUN_LOG"
: > "$GPU_LOG"
echo "timestamp,gpu_index,memory_used_mib,memory_total_mib" >> "$GPU_LOG"

cd "$ROOT"
timeout 300s "$PY" -u examples/ray_parallel_example.py > "$RUN_LOG" 2>&1 &
RUN_PID=$!

PEAK=0
while kill -0 "$RUN_PID" 2>/dev/null; do
  TS=$(date +%s)
  while IFS=, read -r idx used total; do
    idx=$(echo "$idx" | xargs)
    used=$(echo "$used" | xargs)
    total=$(echo "$total" | xargs)
    echo "$TS,$idx,$used,$total" >> "$GPU_LOG"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used > PEAK )); then
      PEAK=$used
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
  sleep 2
done

wait "$RUN_PID"
RUN_EXIT=$?

echo "RUN_EXIT=$RUN_EXIT"
echo "PEAK_GPU_MEMORY_MIB=$PEAK"
if (( PEAK > 10240 )); then
  echo "GPU_MEMORY_CHECK=PASS(>10GiB)"
else
  echo "GPU_MEMORY_CHECK=FAIL(<=10GiB)"
fi

echo "--- tail run log ---"
tail -n 50 "$RUN_LOG"
