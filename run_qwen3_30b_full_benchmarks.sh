#!/bin/bash

# Qwen Full Benchmarks - 99% completion
# Runs on GPUs 2,3 with TP=2 on port 8001
TMP_HOME=/nethome/jkim3934
MODEL_PATH="/data/models_dir/jkim3934/qwen3-omni-30b-thinking"
MODEL_NAME="qwen3"
PORT=8019
TP_SIZE=1
GPUS="2"

# Config files for Qwen
configs=(
    # "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha1_beta1_optimized.json"
    # "${TMP_HOME}/prowl/configs/qwen3_30b/qwen_do-nothing.json"
    # "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha0.7_beta1_optimized.json"
    # "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha0.8_beta1_optimized.json"
    # "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha0.9_beta1_optimized.json"
    # "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha2_beta1_optimized.json"
    # "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha3_beta1_optimized.json"

    # "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha2_beta2_optimized.json"
    # "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha2_beta3_optimized.json"
    "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha3_beta2_optimized.json"
    # "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha3_beta3_optimized.json"
    # "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha3_beta4_optimized.json"
    # "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha3_beta5_optimized.json"
)

# Use 99% completion for all benchmarks
LIMIT_PERCENT=0.99

# List of benchmarks to run
# Using default limits from lm_eval_online_serve.py
benchmarks=("coqa" "coqa")

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
