#!/bin/bash

# Qwen Full Benchmarks - 99% completion
# Runs on GPUs 2,3 with TP=2 on port 8001

MODEL_PATH="Qwen/Qwen3-235B-A22B-Instruct-2507"
MODEL_NAME="qwen3"
PORT=8009
TP_SIZE=4
GPUS="2,3,4,5"

# Config files for Qwen
configs=(
    "${TMP_HOME}/prowl/configs/qwen3/quant_alpha1_beta1_optimized.json"
    "${TMP_HOME}/prowl/configs/qwen3/qwen_do-nothing.json"
)

# Use 99% completion for all benchmarks
LIMIT_PERCENT=0.99

# List of benchmarks to run
# Using default limits from lm_eval_online_serve.py
benchmarks=("humaneval" "minerva_math_algebra")

echo "Starting Qwen full benchmark suite on GPUs ${GPUS}"
echo "Using default limits from lm_eval_online_serve.py"
echo "========================================="

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .json)
    echo "Config: $config_name"
    echo "-----------------------------------------"
    
    for benchmark in "${benchmarks[@]}"; do
        
        echo "Running $benchmark with $config_name"
        
        # Note: run_mixtral_adv_configurable.sh is a generic runner that works for both Mixtral and Qwen
        # Pass empty string for limit to use defaults
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

echo "All Qwen benchmarks completed!"
