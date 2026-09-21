#!/usr/bin/env bash
# Launch LiDAR-only BEV U-Net training on all local GPUs with torchrun.
#
# Usage:
#   ./scripts/run_training.sh <run_name> [extra train_lidar.py args...]
#
# Foreground (holds the terminal, streams output):
#   ./scripts/run_training.sh bev_unet_001 --epochs 20 --wandb
#
# Detached (returns immediately; survives closing the terminal):
#   DETACH=1 RUNS_DIR=/Volumes/daai_ke_team/default/images/lyft_3d_object_detection/runs \
#       ./scripts/run_training.sh bev_unet_001 --epochs 20 --wandb --terminate-cluster
#
# Logging: the live log is written ONLY to node-local disk while training runs
# (streaming small writes to a /Volumes FUSE mount blocks the training
# processes in uninterruptible I/O). When training ends, the log is copied
# once to RUNS_DIR. Checkpoints and metrics are unaffected: the trainer writes
# them straight to RUNS_DIR throughout, using FUSE-safe atomic writes.
#
# --terminate-cluster is handled HERE, not inside the trainer, so the order is
# guaranteed: training ends -> log copied to the Volume -> cluster terminated.
# On a crash, the cluster is only terminated when at least one epoch completed
# (last.pt exists); setup failures leave it running for debugging.
#
# Monitor a detached run:
#   tail -f /local_disk0/run_logs/<run_name>.log        # live local log
#   kill "$(cat /local_disk0/run_logs/<run_name>.pid)"  # stop it (resume later)
#
# Requires: WANDB_API_KEY in the environment (or `wandb login` once on the
# node) when --wandb is passed. When detaching with --wandb, make sure the
# login happened before submission -- the background job cannot prompt.
set -euo pipefail

RUN_NAME="${1:?usage: run_training.sh <run_name> [extra args...]}"
shift

# Intercept --terminate-cluster; everything else is passed to the trainer.
TERMINATE=0
TRAIN_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--terminate-cluster" ]]; then TERMINATE=1; else TRAIN_ARGS+=("$arg"); fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${CACHE_DIR:-/local_disk0/mds_cache}"
RUNS_DIR="${RUNS_DIR:-/local_disk0/runs}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-/local_disk0/run_logs}"
LOG_FILE="$LOCAL_LOG_DIR/$RUN_NAME.log"
PID_FILE="$LOCAL_LOG_DIR/$RUN_NAME.pid"

mkdir -p "$RUNS_DIR" "$LOCAL_LOG_DIR"
cd "$REPO_ROOT"

# Clear leftover shared memory from previously killed runs; a no-op otherwise.
python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()" \
    2>/dev/null || true

COMMAND=(torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/train_lidar.py
         --cache "$CACHE_DIR" --out "$RUNS_DIR/$RUN_NAME" "${TRAIN_ARGS[@]}")

copy_log() {
    if [[ "$LOG_FILE" != "$RUNS_DIR/$RUN_NAME.log" ]]; then
        cp -f "$LOG_FILE" "$RUNS_DIR/$RUN_NAME.log" || true
    fi
}

finalize() {  # copy the log to RUNS_DIR first, only then terminate the cluster
    local status="$1"
    echo "Training exited with status $status; copying log to $RUNS_DIR." >> "$LOG_FILE"
    copy_log
    if [[ "$TERMINATE" == 1 ]]; then
        if [[ "$status" -eq 0 || -f "$RUNS_DIR/$RUN_NAME/last.pt" ]]; then
            python -c "import sys; sys.path.insert(0, '.'); \
from scripts.train_lidar import terminate_cluster; terminate_cluster()" >> "$LOG_FILE" 2>&1 || true
        else
            echo "Training failed during setup (status $status); cluster left running." >> "$LOG_FILE"
        fi
        copy_log  # refresh the snapshot so the termination outcome is captured too
    fi
}

supervise() {  # run training, then finalize; safe to detach
    local status=0
    "${COMMAND[@]}" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    wait $! || status=$?
    finalize "$status"
    return "$status"
}

if [[ "${DETACH:-0}" == "1" ]]; then
    # Progress bars are pointless in a file and generate write churn.
    COMMAND+=(--no-progress)
    supervise > /dev/null 2>&1 &
    disown
    sleep 1
    echo "Submitted run '$RUN_NAME' (torchrun PID $(cat "$PID_FILE")). Terminal is free."
    echo "  live log:  tail -f $LOG_FILE"
    echo "  final log: $RUNS_DIR/$RUN_NAME.log (copied when the run ends)"
    echo "  stop:      kill \$(cat $PID_FILE)"
else
    status=0
    "${COMMAND[@]}" 2>&1 | tee "$LOG_FILE" || status=$?
    finalize "$status"
    exit "$status"
fi
