#!/bin/bash

# Mixtral-8x22B-Instruct Full Benchmarks
# Mixtral-8x22B is a 141B MoE model (top-2 of 8 experts)
# Uses same configs as Mixtral-8x7B (same expert structure)
export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
MODEL_PATH="mistralai/Mixtral-8x22B-Instruct-v0.1"
MODEL_NAME="mixtral_8x22b"
PORT=8021
TP_SIZE=4
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Config files (same as Mixtral-8x7B - same 8 experts, top-2 structure)
configs=(
    # Baseline
    "${TMP_HOME}/prowl/configs/mixtral/do_nothing.json"
    # Prowl
    "${TMP_HOME}/prowl/configs/mixtral/quant_alpha1_beta1_optimized.json"
)

# All benchmarks
benchmarks=("humaneval" "mbpp" "minerva_math_algebra" "gsm8k" "truthfulqa" "sqaud_completion" "longbench_narrativeqa" "coqa" "cnn_dailymail" "hotpotqa" "xsum")

echo "Starting Mixtral-8x22B-Instruct full benchmark suite on GPUs ${GPUS}"
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

echo "All Mixtral-8x22B-Instruct benchmarks completed!"
