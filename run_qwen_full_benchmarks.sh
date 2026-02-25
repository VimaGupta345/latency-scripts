#!/bin/bash

# Qwen Full Benchmarks - 99% completion
# Runs on GPUs 2,3 with TP=2 on port 8001

MODEL_PATH="Qwen/Qwen2-57B-A14B-Instruct"
MODEL_NAME="qwen"
PORT=8089
TP_SIZE=4
GPUS="0,1,2,3"
ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL:-false}

# Config files for Qwen
configs=(
    "${TMP_HOME}/prowl/configs/qwen/quant_alpha1_optimized.json"
    "${TMP_HOME}/prowl/configs/qwen/qwen_do-nothing.json"
)

# Use 99% completion for all benchmarks
LIMIT_PERCENT=0.99

# List of benchmarks to run
# Using default limits from lm_eval_online_serve.py
benchmarks=("gsm8k")
#("gsm8k" "mbpp" "minerva_math_algebra" "humaneval" "hotpotqa" "xsum")

echo "Starting Qwen full benchmark suite on GPUs ${GPUS}"
echo "Using default limits from lm_eval_online_serve.py"
echo "Expert parallelism: ${ENABLE_EXPERT_PARALLEL}"
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
            ${TP_SIZE} \
            16 \
            ${ENABLE_EXPERT_PARALLEL}
        
        echo "Completed: $benchmark with $config_name"
        echo ""
    done
    echo "========================================="
done

echo "All Qwen benchmarks completed!"
