#!/usr/bin/env bash
# Chain several run_training.sh runs back-to-back in ONE detached session.
#
# Usage:
#   DETACH=1 ./scripts/run_chain.sh <chain_name> \
#       -- <run_name> [train args...] \
#       -- <run_name> [train args...] [...]
#
# Each "--"-separated group is passed verbatim to run_training.sh. Runs execute
# strictly sequentially; the chain aborts on the first failure, so a failed run
# leaves the cluster up for debugging / --resume and later runs never start.
#
# --terminate-cluster is only allowed in the FINAL group (anywhere earlier
# would stop the cluster mid-chain); this is validated before submission.
#
# Detach mechanics are identical to run_training.sh: the chain re-execs itself
# via setsid --fork into its own session, so an ssh-tunnel teardown cannot
# signal it. Each run is then executed with DETACH=1 DETACHED_SESSION=1, i.e.
# run_training.sh's supervise() runs directly inside this already-safe session:
# per-run PID files, --no-progress, log publication and (final-run) cluster
# termination all behave exactly as in a single detached run.
#
# Monitor:
#   tail -f /local_disk0/run_logs/<chain_name>.chain.log   # run transitions
#   tail -f /local_disk0/run_logs/<run_name>.log           # current run
# Stop the whole chain (SIGTERMs the current run's torchrun; later runs are
# never started; the cluster stays up):
#   kill "$(cat /local_disk0/run_logs/<chain_name>.chain.pid)"
# Killing only the current run's <run_name>.pid also aborts the chain, since
# the killed run exits non-zero.
#
# Foreground chaining is intentionally unsupported: without DETACH=1 you can
# simply run run_training.sh twice joined by '&&'. This script exists precisely
# for the detached case, where '&&' would die with the ssh session.
set -euo pipefail

LAUNCH_ARGS=("$@")  # kept verbatim so the detached chain can re-exec itself

usage="usage: DETACH=1 run_chain.sh <chain_name> -- <run_name> [args...] [-- <run_name> [args...]]..."
CHAIN_NAME="${1:?$usage}"
shift
[[ "${1:-}" == "--" ]] || { echo "$usage" >&2; exit 1; }
shift
[[ $# -gt 0 ]] || { echo "$usage" >&2; exit 1; }

if [[ "${DETACH:-0}" != "1" ]]; then
    echo "run_chain.sh only supports DETACH=1; for a foreground chain just run" >&2
    echo "run_training.sh sequentially joined by '&&'." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TRAINING="$REPO_ROOT/scripts/run_training.sh"
CACHE_DIR="${CACHE_DIR:-/local_disk0/mds_cache}"
RUNS_DIR="${RUNS_DIR:-/local_disk0/runs}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-/local_disk0/run_logs}"
CHAIN_LOG="$LOCAL_LOG_DIR/$CHAIN_NAME.chain.log"
CHAIN_PID_FILE="$LOCAL_LOG_DIR/$CHAIN_NAME.chain.pid"
mkdir -p "$LOCAL_LOG_DIR"

# Validate the groups before anything is submitted:
#   - every group must be non-empty and start with a run name, not an option;
#   - --terminate-cluster may only appear in the final group.
RUN_NAMES=()
new_group=1
seen_terminate=0
for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        if [[ "$new_group" == 1 ]]; then
            echo "error: empty run group (two '--' in a row?)" >&2; exit 1
        fi
        if [[ "$seen_terminate" == 1 ]]; then
            echo "error: --terminate-cluster in a non-final run would stop the cluster mid-chain" >&2
            exit 1
        fi
        new_group=1
    else
        if [[ "$new_group" == 1 ]]; then
            if [[ "$arg" == -* ]]; then
                echo "error: run group must start with a run name, got '$arg'" >&2; exit 1
            fi
            RUN_NAMES+=("$arg")
            new_group=0
        elif [[ "$arg" == "--terminate-cluster" ]]; then
            seen_terminate=1
        fi
    fi
done
if [[ "$new_group" == 1 ]]; then
    echo "error: trailing '--' with no run group" >&2; exit 1
fi

chain_log() {
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] chain '$CHAIN_NAME': $*" >> "$CHAIN_LOG"
}

supervise_chain() {
    echo $$ > "$CHAIN_PID_FILE"
    chain_log "started (PID $$); runs: ${RUN_NAMES[*]}"

    local stop=0 run_pid_file=""
    on_signal() {
        stop=1
        chain_log "stop requested; signalling current run"
        if [[ -n "$run_pid_file" && -s "$run_pid_file" ]]; then
            kill "$(cat "$run_pid_file")" 2>/dev/null || true
        fi
    }
    trap on_signal TERM INT

    # Sentinel '--' flushes the last group.
    set -- "$@" --
    local group=() arg run_name child status
    for arg in "$@"; do
        if [[ "$arg" != "--" ]]; then group+=("$arg"); continue; fi
        run_name="${group[0]}"
        run_pid_file="$LOCAL_LOG_DIR/$run_name.pid"
        rm -f "$run_pid_file"  # never signal a stale PID from an earlier run

        if [[ "$stop" == 1 ]]; then
            chain_log "stopped before run '$run_name' started"
            exit 130
        fi

        chain_log "starting run '$run_name'"
        DETACH=1 DETACHED_SESSION=1 CACHE_DIR="$CACHE_DIR" RUNS_DIR="$RUNS_DIR" \
            LOCAL_LOG_DIR="$LOCAL_LOG_DIR" NUM_GPUS="$NUM_GPUS" \
            "$RUN_TRAINING" "${group[@]}" &
        child=$!

        # `wait` returns 128+sig when interrupted by the trap without reaping
        # the child; keep waiting until the child's real status is collected.
        status=0
        wait "$child" || status=$?
        while kill -0 "$child" 2>/dev/null; do
            status=0
            wait "$child" || status=$?
        done

        if [[ "$status" -ne 0 ]]; then
            chain_log "run '$run_name' failed (status $status); aborting chain," \
                      "cluster left running. Later runs not started."
            exit "$status"
        fi
        chain_log "run '$run_name' finished successfully"
        group=()
        run_pid_file=""
    done
    chain_log "all runs finished successfully"
}

if [[ "${DETACHED_SESSION:-0}" == "1" ]]; then
    supervise_chain "$@" > /dev/null 2>&1
    exit "$?"
fi

# Re-exec in a new session, with no controlling terminal, so that an sshd or
# ssh-tunnel teardown cannot signal the chain (see run_training.sh for the
# full rationale).
rm -f "$CHAIN_PID_FILE"
DETACH=1 DETACHED_SESSION=1 CACHE_DIR="$CACHE_DIR" RUNS_DIR="$RUNS_DIR" \
    LOCAL_LOG_DIR="$LOCAL_LOG_DIR" NUM_GPUS="$NUM_GPUS" \
    setsid --fork "$0" "${LAUNCH_ARGS[@]}" < /dev/null > /dev/null 2>&1

deadline=$((SECONDS + 30))
while [[ ! -s "$CHAIN_PID_FILE" && "$SECONDS" -lt "$deadline" ]]; do
    sleep 0.2
done
if [[ ! -s "$CHAIN_PID_FILE" ]]; then
    echo "Failed to submit chain '$CHAIN_NAME'; see $CHAIN_LOG." >&2
    exit 1
fi

echo "Submitted chain '$CHAIN_NAME' (PID $(cat "$CHAIN_PID_FILE")): ${RUN_NAMES[*]}. Terminal is free."
echo "  chain log:  tail -f $CHAIN_LOG"
echo "  run logs:   tail -f $LOCAL_LOG_DIR/<run_name>.log"
echo "  stop chain: kill \$(cat $CHAIN_PID_FILE)"
