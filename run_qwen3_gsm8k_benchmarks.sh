#!/bin/bash

# Qwen3 GSM8K Benchmarks
# Runs on GPUs 6,7 with TP=2 on port 8009
#
# Override any variable via env, e.g.:
#   PORT=8070 GPUS="4,5" bash run_qwen3_gsm8k_benchmarks.sh

set -euo pipefail

LOCAL_MODEL_ROOT=${LOCAL_MODEL_ROOT:-/data/models_dir/jkim3934}
MODEL_PATH=${MODEL_PATH:-${LOCAL_MODEL_ROOT}/qwen3-omni-30b-thinking}
MODEL_NAME=${MODEL_NAME:-qwen3}
PORT=${PORT:-8009}
TP_SIZE=${TP_SIZE:-1}
GPUS=${GPUS:-1}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-16}
ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL:-false}

# Qwen3 reasoning-model needs chat-completions path, higher gen cap,
# and the qwen3 reasoning parser on the vLLM server.
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

configs=(
    "${TMP_HOME:-/nethome/jkim3934}/prowl/configs/qwen3/quant_alpha1_optimized.json"
    # "${TMP_HOME:-/nethome/jkim3934}/prowl/configs/qwen3/qwen_do-nothing.json"
)

benchmarks=("gsm8k")

echo "Starting Qwen3 GSM8K benchmark suite on GPU ${GPUS}"
echo "Model: ${MODEL_PATH}"
echo "========================================="

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .json)
    echo "Config: $config_name"
    echo "-----------------------------------------"

    for benchmark in "${benchmarks[@]}"; do
        echo "Running $benchmark with $config_name"

        CUDA_VISIBLE_DEVICES=${GPUS} ./run_mixtral_adv_configurable.sh \
            "${PORT}" \
            "${MODEL_NAME}" \
            "${MODEL_PATH}" \
            "${benchmark}" \
            "" \
            "${config}" \
            "${TP_SIZE}" \
            "${MAX_BATCH_SIZE}" \
            "${ENABLE_EXPERT_PARALLEL}"

        echo "Completed: $benchmark with $config_name"
        echo ""
    done
    echo "========================================="
done

echo "All Qwen3 GSM8K benchmarks completed!"
