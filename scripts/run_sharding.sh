#!/usr/bin/env bash
# Launch MDS shard generation in the background (nohup) and auto-terminate the
# Databricks cluster (stop, NOT delete) when sharding completes successfully.
#
# Usage:
#   bash run_sharding.sh                 # full run: train,val,test + terminate cluster
#   bash run_sharding.sh --no-terminate  # full run, leave cluster up
#   bash run_sharding.sh --limit 20 --splits train --no-terminate   # smoke test
#
# All other flags are forwarded to create_shards.py (see --help).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/shard_$(date +%Y%m%d_%H%M%S).log"

# Pick a Python that has mosaicml-streaming installed (Databricks cluster env first).
PYTHON=""
for candidate in "${PYSPARK_PYTHON:-}" \
                 "${DATABRICKS_ROOT_VIRTUALENV_ENV:-}/bin/python" \
                 /databricks/python/bin/python \
                 python3; do
    if [[ -n "${candidate}" ]] && command -v "${candidate}" >/dev/null 2>&1 \
        && "${candidate}" -c 'import streaming, numpy' >/dev/null 2>&1; then
        PYTHON="${candidate}"
        break
    fi
done
if [[ -z "${PYTHON}" ]]; then
    echo "ERROR: no Python with 'mosaicml-streaming' found." >&2
    exit 1
fi

# --no-terminate opts out of cluster termination; default is to terminate.
TERMINATE_FLAG="--terminate-cluster"
ARGS=()
for arg in "$@"; do
    if [[ "${arg}" == "--no-terminate" ]]; then
        TERMINATE_FLAG=""
    else
        ARGS+=("${arg}")
    fi
done

if [[ -n "${TERMINATE_FLAG}" ]]; then
    for var in DATABRICKS_HOST DATABRICKS_TOKEN DATABRICKS_CLUSTER_ID; do
        if [[ -z "${!var:-}" ]]; then
            echo "ERROR: ${var} is not set; cannot auto-terminate the cluster." >&2
            echo "       Re-run with --no-terminate or export ${var}." >&2
            exit 1
        fi
    done
fi

echo "Python:  ${PYTHON}"
echo "Log:     ${LOG_FILE}"
echo "Args:    ${ARGS[*]:-<defaults>} ${TERMINATE_FLAG}"

nohup "${PYTHON}" -u "${SCRIPT_DIR}/create_shards.py" ${TERMINATE_FLAG} "${ARGS[@]}" \
    > "${LOG_FILE}" 2>&1 &
PID=$!
echo "Started sharding in background: PID ${PID}"
echo "Follow progress with:  tail -f ${LOG_FILE}"
if [[ -n "${TERMINATE_FLAG}" ]]; then
    echo "NOTE: the cluster will TERMINATE automatically when sharding finishes successfully."
fi
