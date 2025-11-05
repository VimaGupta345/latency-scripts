#!/bin/bash

# DeepSeek Full Benchmarks - 99% completion
# Runs on GPU 4 with TP=1 on port 8002

MODEL_PATH="deepseek-ai/DeepSeek-V2-Lite-Chat"
MODEL_NAME="deepseek"
PORT=8002
TP_SIZE=1
GPUS="0"

# Config files for DeepSeek
configs=(
    "${TMP_HOME}/prowl/configs/deepseek/config_do_nothing.json"
    "${TMP_HOME}/prowl/configs/deepseek/deepseek_configs/config_alpha1_beta1.json"
)

# Use 99% completion for all benchmarks
LIMIT_PERCENT=0.99

# List of benchmarks to run
# Using default limits from lm_eval_online_serve.py
benchmarks=("gsm8k")

echo "Starting DeepSeek full benchmark suite on GPU ${GPUS}"
echo "Using default limits from lm_eval_online_serve.py"
echo "========================================="

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .json)
    echo "Config: $config_name"
    echo "-----------------------------------------"
    
    for benchmark in "${benchmarks[@]}"; do
        
        echo "Running $benchmark with $config_name"
        
        # Note: run_mixtral_adv_configurable.sh is a generic runner that works for all models
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

echo "All DeepSeek benchmarks completed!"