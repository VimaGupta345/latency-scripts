#!/bin/bash

# DeepSeek-Coder-V2-Instruct - Prowl config only (baseline already completed)
export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
MODEL_PATH="deepseek-ai/DeepSeek-Coder-V2-Instruct"
MODEL_NAME="deepseek_v2"
PORT=8020
TP_SIZE=4
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Prowl config only
configs=(
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1_beta1_optimized.json"
)

# All benchmarks
benchmarks=("humaneval" "mbpp" "minerva_math_algebra" "gsm8k" "truthfulqa" "sqaud_completion" "longbench_narrativeqa" "coqa" "cnn_dailymail" "hotpotqa" "xsum")

echo "Starting DeepSeek-Coder-V2-Instruct PROWL-ONLY benchmarks on GPUs ${GPUS}"
echo "========================================="

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .json)
    echo "Config: $config_name"
    echo "-----------------------------------------"

    for benchmark in "${benchmarks[@]}"; do
        echo "Running $benchmark with $config_name"

        CUDA_VISIBLE_DEVICES=${GPUS} ./run_mixtral_adv_configurable.sh \
            ${PORT} \
            ${MODEL_NAME} \
            ${MODEL_PATH} \
            ${benchmark} \
            "" \
            ${config} \
            ${TP_SIZE}

        echo "Completed: $benchmark with $config_name"
        echo ""
    done
    echo "========================================="
done

echo "All prowl-only benchmarks completed!"
