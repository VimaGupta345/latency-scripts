#!/bin/bash

# Co-located runs for new alpha/beta configs — 4 benchmarks each
# Models & configs:
#   Qwen2-57B:       baseline + quant_alpha4_beta5_optimized       (TP=2)
#   Qwen3-30B:       baseline + quant_alpha3_beta3_optimized       (TP=1)
#   DeepSeek-V2:     baseline + quant_alpha2_beta4_optimized       (TP=4)
#   Mixtral:         baseline + quant_alpha1_beta1_optimized
#                             + quant_alpha1.4_beta2_optimized     (TP=2)
#
# Usage:
#   bash run_colocated_new_configs.sh [start_from]
#
# Example:
#   bash run_colocated_new_configs.sh      # all from the start
#   bash run_colocated_new_configs.sh 9    # resume from run 9

export TMP_HOME="${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}"
PROWL_ROOT="${TMP_HOME}/prowl"

START_FROM=${1:-1}

LOGDIR="${TMP_HOME}/results/logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/colocated_new_configs_$(date +%Y%m%d-%H%M%S).log"

exec > >(tee -a "${LOGFILE}") 2>&1

benchmarks=("humaneval" "mbpp" "gsm8k" "minerva_math_algebra")

# Total: (2+2+2+3) configs * 4 benchmarks = 36 runs
TOTAL_RUNS=36
RUN_NUM=0

run_benchmarks() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    local PORT=$3
    local TP=$4
    local GPUS=$5
    local CONFIG=$6

    export TP_SIZE="${TP}"
    export CUDA_VISIBLE_DEVICES="${GPUS}"

    local config_name=$(basename "$CONFIG" .json)

    for benchmark in "${benchmarks[@]}"; do
        RUN_NUM=$((RUN_NUM + 1))

        if [ ${RUN_NUM} -lt ${START_FROM} ]; then
            echo "  [${RUN_NUM}/${TOTAL_RUNS}] ${MODEL_NAME} / ${benchmark} / ${config_name} — SKIPPED (before start_from=${START_FROM})"
            continue
        fi

        echo ""
        echo "  [${RUN_NUM}/${TOTAL_RUNS}] ${MODEL_NAME} / ${benchmark} / ${config_name}"
        echo "  $(date)"

        ./run_mixtral_adv_configurable.sh \
            ${PORT} \
            ${MODEL_NAME} \
            ${MODEL_PATH} \
            ${benchmark} \
            "" \
            ${CONFIG} \
            ${TP}

        echo "  Done: ${MODEL_NAME} / ${benchmark} / ${config_name}"
    done
}

echo "========================================="
echo "CO-LOCATED NEW CONFIGS — 4 models, 4 benchmarks"
echo "  Total runs:  ${TOTAL_RUNS}"
echo "  Log:         ${LOGFILE}"
echo "  Started:     $(date)"
echo "========================================="

# =========================================
# Qwen2-57B: TP=2, GPUs 0,1, port 8030
# =========================================
echo ""
echo "========================================="
echo "MODEL: qwen (Qwen/Qwen2-57B-A14B-Instruct)"
echo "  Port: 8030, TP: 2, GPUs: 0,1"
echo "========================================="

run_benchmarks qwen \
    Qwen/Qwen2-57B-A14B-Instruct \
    8030 2 "0,1" \
    "${PROWL_ROOT}/configs/qwen/qwen_do-nothing.json"

run_benchmarks qwen \
    Qwen/Qwen2-57B-A14B-Instruct \
    8030 2 "0,1" \
    "${PROWL_ROOT}/configs/qwen/quant_alpha4_beta5_optimized.json"

# =========================================
# Qwen3-30B: TP=1, GPU 0, port 8019
# =========================================
echo ""
echo "========================================="
echo "MODEL: qwen3 (Qwen/Qwen3-30B-A3B-Instruct-2507)"
echo "  Port: 8019, TP: 1, GPUs: 0"
echo "========================================="

run_benchmarks qwen3 \
    Qwen/Qwen3-30B-A3B-Instruct-2507 \
    8019 1 "0" \
    "${PROWL_ROOT}/configs/qwen3_30b/qwen_do-nothing.json"

run_benchmarks qwen3 \
    Qwen/Qwen3-30B-A3B-Instruct-2507 \
    8019 1 "0" \
    "${PROWL_ROOT}/configs/qwen3_30b/quant_alpha3_beta3_optimized.json"

# =========================================
# DeepSeek-V2-Coder: TP=4, GPUs 0,1,2,3, port 8020
# =========================================
echo ""
echo "========================================="
echo "MODEL: deepseek_v2 (deepseek-ai/DeepSeek-Coder-V2-Instruct)"
echo "  Port: 8020, TP: 4, GPUs: 0,1,2,3"
echo "========================================="

run_benchmarks deepseek_v2 \
    deepseek-ai/DeepSeek-Coder-V2-Instruct \
    8020 4 "0,1,2,3" \
    "${PROWL_ROOT}/configs/deepseek_v2_coder/config_do_nothing.json"

run_benchmarks deepseek_v2 \
    deepseek-ai/DeepSeek-Coder-V2-Instruct \
    8020 4 "0,1,2,3" \
    "${PROWL_ROOT}/configs/deepseek_v2_coder/quant_alpha2_beta4_optimized.json"

# =========================================
# Mixtral: TP=2, GPUs 0,1, port 8040
# (2 prowl configs + 1 baseline = 3 config sweeps)
# =========================================
echo ""
echo "========================================="
echo "MODEL: mixtral (mistralai/Mixtral-8x7B-Instruct-v0.1)"
echo "  Port: 8040, TP: 2, GPUs: 0,1"
echo "========================================="

run_benchmarks mixtral \
    mistralai/Mixtral-8x7B-Instruct-v0.1 \
    8040 2 "0,1" \
    "${PROWL_ROOT}/configs/mixtral/do_nothing.json"

run_benchmarks mixtral \
    mistralai/Mixtral-8x7B-Instruct-v0.1 \
    8040 2 "0,1" \
    "${PROWL_ROOT}/configs/mixtral/quant_alpha1_beta1_optimized.json"

run_benchmarks mixtral \
    mistralai/Mixtral-8x7B-Instruct-v0.1 \
    8040 2 "0,1" \
    "${PROWL_ROOT}/configs/mixtral/quant_alpha1.4_beta2_optimized.json"

echo ""
echo "========================================="
echo "ALL CO-LOCATED RUNS COMPLETED"
echo "  Finished: $(date)"
echo "  Total:    ${RUN_NUM}/${TOTAL_RUNS} attempted"
echo "  Log:      ${LOGFILE}"
echo "========================================="
