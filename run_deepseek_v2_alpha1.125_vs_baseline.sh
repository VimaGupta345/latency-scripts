#!/bin/bash

# DeepSeek-Coder-V2-Instruct: alpha1.125_beta2 vs baseline
# Compare the safest prowl config against do_nothing baseline

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
MODEL_PATH="deepseek-ai/DeepSeek-Coder-V2-Instruct"
MODEL_NAME="deepseek_v2"
PORT=8020
TP_SIZE=4
GPUS="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

configs=(
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/config_do_nothing.json"
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1.125_beta2_optimized.json"
)

benchmarks=("humaneval" "mbpp" "gsm8k" "minerva_math_algebra")

echo "Starting DeepSeek-Coder-V2: alpha1.125_beta2 vs baseline on GPUs ${GPUS}"
echo "Benchmarks: ${benchmarks[*]}"
echo "========================================="

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .json)
    echo ""
    echo "Config: $config_name"
    echo "-----------------------------------------"

    for benchmark in "${benchmarks[@]}"; do
        echo "  Running $benchmark with $config_name"

        CUDA_VISIBLE_DEVICES=${GPUS} ./run_mixtral_adv_configurable.sh \
            ${PORT} \
            ${MODEL_NAME} \
            ${MODEL_PATH} \
            ${benchmark} \
            "" \
            ${config} \
            ${TP_SIZE}

        echo "  Completed: $benchmark with $config_name"
    done
    echo "-----------------------------------------"
done

echo ""
echo "========================================="
echo "Experiment completed! Run gather_results.py to compare:"
echo "  python gather_results.py results/DeepSeek-Coder-V2-Instruct/"
echo "========================================="
