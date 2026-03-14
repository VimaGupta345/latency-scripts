#!/bin/bash

# DeepSeek-Coder-V2-Instruct Alpha/Beta Sweep
# Goal: Find alpha/beta where all benchmarks stay within 1% accuracy of baseline
# Model: 236B MoE (top-6 of 160 experts), TP=4
#
# Baseline results (do_nothing):
#   humaneval:           0.7683 (pass@1)      | 29.93 tok/s
#   mbpp:                0.7880 (pass_at_1)    | 26.37 tok/s
#   gsm8k:               0.5880 (exact_match)  | 30.06 tok/s
#   minerva_math_algebra: 0.1120 (exact_match)  | 33.12 tok/s
#
# alpha=1 beta=2(hardcoded) results:
#   humaneval: 0.7683 (0%     delta), 31.64 tok/s (+5.7%)
#   mbpp:      0.7960 (+0.8%  delta), 28.71 tok/s (+8.9%)
#   gsm8k:     0.5760 (-1.2%  delta), 34.39 tok/s (+14.3%)  <-- over budget
#   math:      0.0880 (-2.4%  delta), 38.69 tok/s (+16.9%)  <-- over budget
#
# Sweep plan: increase alpha for finer bins, use explicit beta

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
MODEL_PATH="deepseek-ai/DeepSeek-Coder-V2-Instruct"
MODEL_NAME="deepseek_v2"
PORT=8020
TP_SIZE=4
GPUS="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

# Configs ordered from most likely sweet spot to most conservative
configs=(
    # alpha=1.25, beta=2: slightly finer bins than baseline (most likely sweet spot)
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1.25_beta2_optimized.json"
    # alpha=1.5, beta=2: finer bins, moderate beta
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1.5_beta2_optimized.json"
    # alpha=1.5, beta=3: finer bins, conservative beta
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1.5_beta3_optimized.json"
    # alpha=2, beta=3: very fine bins, conservative beta (accuracy floor)
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha2_beta3_optimized.json"
)

# Focus benchmarks: the 4 we care about
benchmarks=("humaneval" "mbpp" "gsm8k" "minerva_math_algebra")

echo "Starting DeepSeek-Coder-V2 Alpha/Beta Sweep on GPUs ${GPUS}"
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
echo "Alpha/beta sweep completed! Run gather_results.py to compare:"
echo "  python gather_results.py results/DeepSeek-Coder-V2-Instruct/"
echo "========================================="
