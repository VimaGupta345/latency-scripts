#!/bin/bash
# Run humaneval do-nothing TWICE for each model to warm up Triton/CUDA caches.
# The first run has cold-start overhead (JIT compilation, handshake, etc.)
# The second run gives clean baseline metrics.
#
# Usage:
#   bash run_disagg_warmup_humaneval.sh [prefill_gpus] [decode_gpus]

set -e

export TMP_HOME="${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}"
PROWL_ROOT="${TMP_HOME}/prowl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "${PROWL_ROOT}/.venv/bin/activate"

PREFILL_GPUS=${1:-0,1,2,3}
DECODE_GPUS=${2:-4,5,6,7}

PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8000

# Models: name|path|tp_size|do_nothing_config
MODELS=(
    "qwen2|Qwen/Qwen2-57B-A14B-Instruct|2|${PROWL_ROOT}/configs/qwen/qwen_do-nothing.json"
    "qwen3|Qwen/Qwen3-30B-A3B-Instruct-2507|2|${PROWL_ROOT}/configs/qwen3_30b/qwen_do-nothing.json"
    "mixtral|mistralai/Mixtral-8x7B-Instruct-v0.1|2|${PROWL_ROOT}/configs/mixtral/do_nothing.json"
    "deepseek_v2|deepseek-ai/DeepSeek-Coder-V2-Instruct|4|${PROWL_ROOT}/configs/deepseek_v2_coder/config_do_nothing.json"
)

# Force-kill anything on our ports AND orphaned GPU processes
kill_all() {
    for port in ${PREFILL_PORT} ${DECODE_PORT} ${PROXY_PORT}; do
        for pid in $(lsof -ti:${port} 2>/dev/null); do
            local sid=$(ps -o sid= -p ${pid} 2>/dev/null | tr -d ' ')
            if [ -n "${sid}" ] && [ "${sid}" != "0" ]; then
                kill -9 -${sid} 2>/dev/null || true
            fi
            kill -9 ${pid} 2>/dev/null || true
        done
    done
    local my_uid=$(id -u)
    for gpu_pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        gpu_pid=$(echo "$gpu_pid" | tr -d ' ')
        local owner=$(ps -o uid= -p "$gpu_pid" 2>/dev/null | tr -d ' ')
        if [ "$owner" = "$my_uid" ]; then
            kill -9 "$gpu_pid" 2>/dev/null || true
        fi
    done
    sleep 5
}

echo "========================================="
echo "WARMUP: humaneval do-nothing x2 for each model"
echo "  Prefill GPUs: ${PREFILL_GPUS}"
echo "  Decode GPUs:  ${DECODE_GPUS}"
echo "  Started:      $(date)"
echo "========================================="

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME MODEL_PATH TP_SIZE CONFIG_FILE <<< "${model_entry}"

    echo ""
    echo "========================================="
    echo "MODEL: ${MODEL_NAME} (TP=${TP_SIZE})"
    echo "========================================="

    for run in 1 2; do
        echo ""
        echo "  --- Run ${run}/2: ${MODEL_NAME} humaneval do-nothing ---"
        echo "  $(date)"

        # Clean up before each run
        kill_all

        bash "${SCRIPT_DIR}/run_disagg_lynx.sh" \
            "${MODEL_PATH}" \
            "${CONFIG_FILE}" \
            "${TP_SIZE}" \
            "${PREFILL_GPUS}" \
            "${DECODE_GPUS}" \
            humaneval

        echo "  Run ${run}/2 DONE: $(date)"
    done
done

# Final cleanup
kill_all

echo ""
echo "========================================="
echo "WARMUP COMPLETE: $(date)"
echo "========================================="
