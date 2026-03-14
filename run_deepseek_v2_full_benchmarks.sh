#!/bin/bash

# DeepSeek-Coder-V2-Instruct Full Benchmarks - 99% completion
# DeepSeek-Coder-V2-Instruct is a 236B MoE model (top-6 of 160 experts)
export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
MODEL_PATH="deepseek-ai/DeepSeek-Coder-V2-Instruct"
MODEL_NAME="deepseek_v2"
PORT=8020
TP_SIZE=4
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Config files for DeepSeek-Coder-V2 (160 routed experts, 6 per token)
configs=(
    # Without prowl (baseline - keeps all experts)
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/config_do_nothing.json"
    # With prowl
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1_beta1_optimized.json"
#    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha2_beta1_optimized.json"
#    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha3_beta2_optimized.json"
)

# Use 99% completion for all benchmarks
LIMIT_PERCENT=0.99

# List of benchmarks to run
# Using default limits from lm_eval_online_serve.py
benchmarks=("humaneval" "mbpp" "minerva_math_algebra" "gsm8k" "truthfulqa" "sqaud_completion" "longbench_narrativeqa" "coqa" "cnn_dailymail" "hotpotqa" "xsum")

echo "Starting DeepSeek-Coder-V2-Instruct full benchmark suite on GPUs ${GPUS}"
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

echo "All DeepSeek-Coder-V2-Instruct benchmarks completed!"
