#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TMP_HOME=${TMP_HOME:-/nethome/rdudala3}
RUN_USER=${USER:-$(id -un)}

MODEL_PATH=${MODEL_PATH:-${1:-zai-org/GLM-4.7}}
MODEL_NAME=${MODEL_NAME:-glm-4.7}
BENCHMARK=${BENCHMARK:-swebench}
LIMIT=${LIMIT:-}
PORT=${PORT:-8097}
TP_SIZE=${TP_SIZE:-4}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-16}
GPUS=${GPUS:-0,1,2,3}
GPUS_CLEAN=${GPUS// /}
ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL:-false}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
SWE_BENCH_MAX_TOKENS=${SWE_BENCH_MAX_TOKENS:-4096}
STATS_ROOT=${STATS_ROOT:-/nethome/rdudala3/latency-scripts}
MODEL_CACHE_ROOT=${MODEL_CACHE_ROOT:-/data/models_dir/${RUN_USER}}

CONFIG_DO_NOTHING="${TMP_HOME}/prowl/configs/glm-4.7/glm-do-nothing.json"
CONFIG_QUANT="${TMP_HOME}/prowl/configs/quant_alpha1_optimized.json"
if [ ! -f "${CONFIG_QUANT}" ]; then
    CONFIG_QUANT="${TMP_HOME}/prowl/configs/glm-4.7/quant_alpha1_optimized.json"
fi

configs=(
    "${CONFIG_DO_NOTHING}"
    "${CONFIG_QUANT}"
)

export PERF_STATS_ROOT="${STATS_ROOT}/perf"
export QUALITY_STATS_ROOT="${STATS_ROOT}/quality"
export HF_HOME=${HF_HOME:-"${MODEL_CACHE_ROOT}"}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"${MODEL_CACHE_ROOT}/huggingface"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"${MODEL_CACHE_ROOT}/transformers"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"${MODEL_CACHE_ROOT}/datasets"}
export MAX_MODEL_LEN
export SWE_BENCH_MAX_TOKENS

mkdir -p "${PERF_STATS_ROOT}" "${QUALITY_STATS_ROOT}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

cd "${SCRIPT_DIR}"

echo "Starting GLM-4.7 sweep on benchmark '${BENCHMARK}'"
echo "Model: ${MODEL_PATH}"
echo "Configs:"
printf '  - %s\n' "${configs[@]}"
echo "TP size: ${TP_SIZE}"
echo "CUDA_VISIBLE_DEVICES: ${GPUS_CLEAN}"
echo "Max model len: ${MAX_MODEL_LEN}"
echo "SWE-bench max tokens: ${SWE_BENCH_MAX_TOKENS}"
echo "Stats root: ${STATS_ROOT}"
echo "HF cache: ${HUGGINGFACE_HUB_CACHE}"
echo "Transformers cache: ${TRANSFORMERS_CACHE}"
echo "Datasets cache: ${HF_DATASETS_CACHE}"
echo "========================================="

for config in "${configs[@]}"; do
    if [ ! -f "${config}" ]; then
        echo "Missing config file: ${config}"
        exit 1
    fi

    config_name=$(basename "${config}" .json)
    echo "Running ${BENCHMARK} with ${config_name} ..."

    CUDA_VISIBLE_DEVICES="${GPUS_CLEAN}" ./run_mixtral_adv_configurable.sh \
        "${PORT}" \
        "${MODEL_NAME}" \
        "${MODEL_PATH}" \
        "${BENCHMARK}" \
        "${LIMIT}" \
        "${config}" \
        "${TP_SIZE}" \
        "${MAX_BATCH_SIZE}" \
        "${ENABLE_EXPERT_PARALLEL}"

    echo "Completed ${config_name}"
    echo "-----------------------------------------"
done

echo "All runs complete."
echo "Perf logs: ${PERF_STATS_ROOT}"
echo "Quality logs: ${QUALITY_STATS_ROOT}"
