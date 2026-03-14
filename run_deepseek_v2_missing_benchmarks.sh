#!/bin/bash

# DeepSeek-Coder-V2-Instruct - Re-run failed benchmarks (missing deps now installed)
export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
MODEL_PATH="deepseek-ai/DeepSeek-Coder-V2-Instruct"
MODEL_NAME="deepseek_v2"
PORT=8020
TP_SIZE=4
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

configs=(
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/config_do_nothing.json"
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1_beta1_optimized.json"
)

# Only the benchmarks that failed due to missing packages
benchmarks=("minerva_math_algebra" "longbench_narrativeqa" "cnn_dailymail" "hotpotqa")

echo "Starting DeepSeek-Coder-V2-Instruct MISSING benchmarks on GPUs ${GPUS}"
echo "Benchmarks: ${benchmarks[*]}"
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

echo "All missing benchmarks completed!"
