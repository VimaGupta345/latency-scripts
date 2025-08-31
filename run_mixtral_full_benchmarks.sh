#!/bin/bash

# Mixtral Full Benchmarks - 99% completion
# Runs on GPUs 0,1 with TP=2 on port 8000

MODEL_PATH="/scratch/vgupta345/models_dir/Mixtral-8x7B-Instruct-v0.1/"
MODEL_NAME="mixtral"
PORT=8000
TP_SIZE=2
GPUS="0,1"

# Config files for Mixtral
configs=(
    "${TMP_HOME}/prowl/configs/mixtral/do_nothing.json"
    "${TMP_HOME}/prowl/configs/mixtral/advanced_alpha0_beta0.7.json"
    "${TMP_HOME}/prowl/configs/mixtral/advanced_alpha0_beta1.25.json"
)

# Use 99% completion for all benchmarks
LIMIT_PERCENT=0.99

# List of benchmarks to run
# Using default limits from lm_eval_online_serve.py
benchmarks=("humaneval" "gsm8k" "mbpp" "minerva_math_algebra")

echo "Starting Mixtral full benchmark suite on GPUs ${GPUS}"
echo "Using default limits from lm_eval_online_serve.py"
echo "========================================="

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .json)
    echo "Config: $config_name"
    echo "-----------------------------------------"
    
    for benchmark in "${benchmarks[@]}"; do
        
        echo "Running $benchmark with $config_name"
        
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

echo "All Mixtral benchmarks completed!"