#!/bin/bash

# DeepSeek-Coder-V2-Instruct Alpha/Beta Sweep Round 3
# Goal: Split the difference between α=1.25,β=2 and α=1.25,β=3

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
MODEL_PATH="deepseek-ai/DeepSeek-Coder-V2-Instruct"
MODEL_NAME="deepseek_v2"
PORT=8020
TP_SIZE=4
GPUS="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

configs=(
    # Between the winner (α=1.125) and α=1.25 — aiming for more throughput
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1.175_beta2_optimized.json"
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1.2_beta2_optimized.json"
    # Split β between the α=1.25 configs that straddled the budget
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1.25_beta2.5_optimized.json"
)

benchmarks=("humaneval" "mbpp" "gsm8k" "minerva_math_algebra")

echo "Starting DeepSeek-Coder-V2 Alpha/Beta Sweep Round 3 on GPUs ${GPUS}"
echo "Benchmarks: ${benchmarks[*]}"
echo "========================================="

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .json)
    echo ""
    echo "Config: $config_name ($(cat "$config" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'alpha={d[\"alpha\"]}, beta={d[\"beta\"]}')"))"
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
echo "Sweep round 3 completed!"
echo "========================================="
