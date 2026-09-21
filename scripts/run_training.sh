#!/usr/bin/env bash
# Launch LiDAR-only BEV U-Net training on all local GPUs with torchrun.
#
# Usage:
#   ./scripts/run_training.sh <run_name> [extra train_lidar.py args...]
#
# Foreground (holds the terminal, streams output):
#   ./scripts/run_training.sh bev_unet_001 --epochs 20 --batch-size 4 --wandb
#
# Detached (returns immediately; survives closing the terminal):
#   DETACH=1 RUNS_DIR=/Volumes/daai_ke_team/default/images/lyft_3d_object_detection/runs \
#       ./scripts/run_training.sh bev_unet_001 \
#       --epochs 20 --batch-size 4 --wandb --terminate-cluster
#
# Monitor a detached run:
#   tail -f <RUNS_DIR>/<run_name>.log        # live log
#   kill "$(cat <RUNS_DIR>/<run_name>.pid)"  # stop it (checkpoints/resume remain)
#
# Requires: WANDB_API_KEY in the environment (or `wandb login` once on the
# node) when --wandb is passed. When detaching with --wandb, make sure the
# login happened before submission -- the background job cannot prompt.
set -euo pipefail

RUN_NAME="${1:?usage: run_training.sh <run_name> [extra args...]}"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${CACHE_DIR:-/local_disk0/mds_cache}"
RUNS_DIR="${RUNS_DIR:-/local_disk0/runs}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}"
LOG_FILE="$RUNS_DIR/$RUN_NAME.log"

mkdir -p "$RUNS_DIR"
cd "$REPO_ROOT"

COMMAND=(torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/train_lidar.py
         --cache "$CACHE_DIR" --out "$RUNS_DIR/$RUN_NAME" "$@")

if [[ "${DETACH:-0}" == "1" ]]; then
    nohup "${COMMAND[@]}" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$RUNS_DIR/$RUN_NAME.pid"
    echo "Submitted run '$RUN_NAME' (PID $PID). Terminal is free."
    echo "  log:  tail -f $LOG_FILE"
    echo "  stop: kill $PID"
else
    "${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
fi
