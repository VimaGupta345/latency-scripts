#!/bin/bash

# DeepSeek-Coder-V2-Instruct Alpha/Beta Sweep Round 2
# Goal: Thread the needle between α=1.0 (good MBPP) and α=1.25 (good GSM8K/Math)
#
# Round 1 results (acc delta vs baseline / throughput gain):
#   Config          | HumanEval    | MBPP         | GSM8K        | Math
#   α=1.0,  β=2     | +0.0%/+5.7%  | +0.8%/+8.9%  | -1.2%/+14.4% | -2.4%/+16.8%
#   α=1.25, β=2     | +3.0%/+8.6%  | -1.6%/+7.2%  | -0.4%/+13.3% | +0.0%/+16.0%

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
MODEL_PATH="deepseek-ai/DeepSeek-Coder-V2-Instruct"
MODEL_NAME="deepseek_v2"
PORT=8020
TP_SIZE=4
GPUS="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

configs=(
    # α=1.125, β=2: interpolate between the two best configs
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1.125_beta2_optimized.json"
    # α=1.0, β=3: same alpha that works for MBPP, higher beta to help Math/GSM8K
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1.0_beta3_optimized.json"
    # α=1.25, β=3: best alpha for Math, more conservative tail for MBPP
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1.25_beta3_optimized.json"
)

benchmarks=("humaneval" "mbpp" "gsm8k" "minerva_math_algebra")

echo "Starting DeepSeek-Coder-V2 Alpha/Beta Sweep Round 2 on GPUs ${GPUS}"
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
echo "Sweep round 2 completed! Run gather_results.py to compare:"
echo "  python gather_results.py results/DeepSeek-Coder-V2-Instruct/"
echo "========================================="
