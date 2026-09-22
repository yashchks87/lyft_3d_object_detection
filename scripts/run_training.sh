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
# once to RUNS_DIR. Checkpoints and metrics land in RUNS_DIR too, but the
# trainer stages them on node-local disk and publishes them from a background
# thread, so a slow Volume can never stall the DDP ranks into an NCCL timeout.
#
# --terminate-cluster is handled HERE, not inside the trainer, so the order is
# guaranteed: training ends -> log copied to the Volume -> cluster terminated.
# The cluster is terminated ONLY after a fully successful run (exit status 0);
# any failure leaves it running for debugging and a --resume relaunch.
#
# A detached run is re-executed in a brand new session (setsid), NOT merely
# disowned. The Databricks ssh-tunnel that serves the IDE terminal owns an sshd
# session and shuts itself down once no client has been attached for its
# --shutdown-delay ("No SSH clients for 10m0s, shutting down..."); that teardown
# SIGTERMs every process still in the session. `disown` only suppresses SIGHUP,
# so it does not help -- it is what let a 20-epoch run die in its final epoch,
# ~200ms after the tunnel gave up. A new session has its own process group and
# reparents to init, so no teardown can reach it.
#
# Monitor a detached run:
#   tail -f /local_disk0/run_logs/<run_name>.log        # live local log
#   kill "$(cat /local_disk0/run_logs/<run_name>.pid)"  # stop it (resume later)
#
# Requires: WANDB_API_KEY in the environment (or `wandb login` once on the
# node) when --wandb is passed. When detaching with --wandb, make sure the
# login happened before submission -- the background job cannot prompt.
set -euo pipefail

LAUNCH_ARGS=("$@")  # kept verbatim so a detached run can re-exec itself

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

COMMAND=(torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/train_lidar.py
         --cache "$CACHE_DIR" --out "$RUNS_DIR/$RUN_NAME" "${TRAIN_ARGS[@]}")

# Clear leftover shared memory from previously killed runs; a no-op otherwise.
# Only the process that actually starts training does this: importing streaming
# costs seconds, and the detached launcher re-execs itself, so doing it up front
# would pay that cost twice and delay the PID file the submitter waits for.
clear_stale_shared_memory() {
    python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()" \
        2>/dev/null || true
}

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
        if [[ "$status" -eq 0 ]]; then
            python -c "import sys; sys.path.insert(0, '.'); \
from scripts.train_lidar import terminate_cluster; terminate_cluster()" >> "$LOG_FILE" 2>&1 || true
        else
            echo "Training failed (status $status); cluster left running for debugging/resume." >> "$LOG_FILE"
        fi
        copy_log  # refresh the snapshot so the termination outcome is captured too
    fi
}

supervise() {  # run training, then finalize; safe to detach
    local status=0
    clear_stale_shared_memory
    "${COMMAND[@]}" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    wait $! || status=$?
    finalize "$status"
    return "$status"
}

if [[ "${DETACH:-0}" == "1" ]]; then
    # Progress bars are pointless in a file and generate write churn.
    COMMAND+=(--no-progress)

    if [[ "${DETACHED_SESSION:-0}" == "1" ]]; then
        # Already re-executed into our own session by the branch below: this is
        # the process that actually supervises training.
        status=0
        supervise > /dev/null 2>&1 || status=$?
        exit "$status"
    fi

    # Re-exec in a new session, with no controlling terminal, so that an sshd or
    # ssh-tunnel teardown cannot signal the run. setsid --fork returns at once
    # and the run reparents to init.
    # The already-resolved paths are passed explicitly so the re-exec cannot
    # disagree with what is reported below (the caller may have set them without
    # exporting them, in which case the child would fall back to the defaults).
    rm -f "$PID_FILE"  # never report a stale PID from an earlier run
    DETACH=1 DETACHED_SESSION=1 CACHE_DIR="$CACHE_DIR" RUNS_DIR="$RUNS_DIR" \
        LOCAL_LOG_DIR="$LOCAL_LOG_DIR" NUM_GPUS="$NUM_GPUS" \
        setsid --fork "$0" "${LAUNCH_ARGS[@]}" < /dev/null > /dev/null 2>&1

    # The new session writes the PID file itself, but only after clearing stale
    # shared memory (importing streaming takes seconds), so wait generously.
    deadline=$((SECONDS + 120))
    while [[ ! -s "$PID_FILE" && "$SECONDS" -lt "$deadline" ]]; do
        sleep 0.2
    done
    if [[ ! -s "$PID_FILE" ]]; then
        echo "Failed to submit run '$RUN_NAME'; see $LOG_FILE." >&2
        exit 1
    fi

    echo "Submitted run '$RUN_NAME' (torchrun PID $(cat "$PID_FILE")). Terminal is free."
    echo "  live log:  tail -f $LOG_FILE"
    echo "  final log: $RUNS_DIR/$RUN_NAME.log (copied when the run ends)"
    echo "  stop:      kill \$(cat $PID_FILE)"
else
    status=0
    clear_stale_shared_memory
    "${COMMAND[@]}" 2>&1 | tee "$LOG_FILE" || status=$?
    finalize "$status"
    exit "$status"
fi
