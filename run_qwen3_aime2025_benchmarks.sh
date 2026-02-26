#!/bin/bash
# cd /nethome/jkim3934/latency-scripts
# PYTHON_BIN=/nethome/jkim3934/prowl/.venv/bin/python bash run_qwen3_aime2025_benchmarks.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

TMP_HOME=${TMP_HOME:-/nethome/jkim3934}
PROWL_ROOT=${PROWL_ROOT:-${TMP_HOME}/prowl}
PYTHON_BIN=${PYTHON_BIN:-${PROWL_ROOT}/.venv/bin/python}
if [ ! -x "${PYTHON_BIN}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v python)
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v python3)
    else
        echo "Could not find a usable Python interpreter."
        exit 127
    fi
fi
export PYTHON_BIN
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

if ! "${PYTHON_BIN}" -c "import vllm" >/dev/null 2>&1; then
    echo "Selected Python cannot import vllm: ${PYTHON_BIN}"
    echo "If vLLM works from prowl, use:"
    echo "  PYTHON_BIN=${TMP_HOME}/prowl/.venv/bin/python $0"
    exit 1
fi

if ! command -v lm-eval >/dev/null 2>&1; then
    echo "lm-eval not found on PATH for interpreter: ${PYTHON_BIN}"
    echo "PATH currently starts with: $(dirname "${PYTHON_BIN}")"
    exit 1
fi

LOCAL_MODEL_ROOT=${LOCAL_MODEL_ROOT:-/data/models_dir/jkim3934}
if [ ! -d "${LOCAL_MODEL_ROOT}" ] && [ -d "/data/models-dir/jkim3934" ]; then
    LOCAL_MODEL_ROOT="/data/models-dir/jkim3934"
fi

# Preferred local model for this script (qwen3-omni-30b-thinking at your model dir).
MODEL_PATH=${MODEL_PATH:-${LOCAL_MODEL_ROOT}/qwen3-omni-30b-thinking}
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    ALT_MODEL_PATH="${LOCAL_MODEL_ROOT}/qwen3-30b-thinking"
    if [ -f "${ALT_MODEL_PATH}/config.json" ]; then
        echo "Requested model path not found: ${MODEL_PATH}"
        echo "Using available local model instead: ${ALT_MODEL_PATH}"
        MODEL_PATH="${ALT_MODEL_PATH}"
    else
        echo "Could not find local model at:"
        echo "  ${MODEL_PATH}"
        echo "or fallback:"
        echo "  ${ALT_MODEL_PATH}"
        echo "Set MODEL_PATH explicitly to your model directory (e.g. MODEL_PATH=/data/models_dir/jkim3934/qwen3-omni-30b-thinking)."
        exit 1
    fi
fi

MODEL_NAME=${MODEL_NAME:-qwen3}
PORT=${PORT:-8009}
TP_SIZE=${TP_SIZE:-4}
GPUS=${GPUS:-0,1,2,3}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-16}
ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL:-false}

# lm_eval task name (use aime25 to match lm_eval --tasks aime25). Override with BENCHMARK=... if needed.
BENCHMARK=${BENCHMARK:-aime25}
# Empty LIMIT uses benchmark defaults in lm_eval_online_serve.py.
LIMIT=${LIMIT:-}
# Align with official Qwen AIME guidance:
# - Standard Qwen3: 38912 output tokens
# - Thinking-2507: 81920 output tokens
if [ -z "${MAX_MODEL_LEN:-}" ]; then
    model_lc=$(printf '%s' "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]')
    if [[ "${model_lc}" == *"thinking-2507"* ]]; then
        export MAX_MODEL_LEN=81920
    else
        export MAX_MODEL_LEN=38912
    fi
fi

configs=(
    "${TMP_HOME}/prowl/configs/qwen3/quant_alpha3_beta1_optimized.json"
    "${TMP_HOME}/prowl/configs/qwen3/quant_alpha3_beta2_optimized.json"
    "${TMP_HOME}/prowl/configs/qwen3/qwen_do-nothing.json"
)

echo "Starting Qwen3 benchmark suite"
echo "Benchmark: ${BENCHMARK}"
echo "Model: ${MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${GPUS}"
echo "Port: ${PORT}"
echo "TP size: ${TP_SIZE}"
echo "Max batch size: ${MAX_BATCH_SIZE}"
echo "Max model length: ${MAX_MODEL_LEN}"
echo "Expert parallelism: ${ENABLE_EXPERT_PARALLEL}"
echo "========================================="

for config in "${configs[@]}"; do
    if [ ! -f "${config}" ]; then
        echo "Missing config file: ${config}"
        exit 1
    fi

    config_name=$(basename "${config}" .json)
    echo "Running ${BENCHMARK} with ${config_name}"

    # Uses run_mixtral_adv_configurable.sh, which launches ngram serving.
    CUDA_VISIBLE_DEVICES="${GPUS}" ./run_mixtral_adv_configurable.sh \
        "${PORT}" \
        "${MODEL_NAME}" \
        "${MODEL_PATH}" \
        "${BENCHMARK}" \
        "${LIMIT}" \
        "${config}" \
        "${TP_SIZE}" \
        "${MAX_BATCH_SIZE}" \
        "${ENABLE_EXPERT_PARALLEL}"

    echo "Completed: ${BENCHMARK} with ${config_name}"
    echo "-----------------------------------------"
done

echo "All Qwen3 ${BENCHMARK} benchmarks completed!"
