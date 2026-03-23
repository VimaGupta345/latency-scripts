#!/usr/bin/env zsh

# Run all 4 models x 4 benchmarks SEQUENTIALLY (no GPU contention)
# Usage: zsh run_all_sequential.sh
# Each model runs alone on its required GPUs.

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source

LOGDIR="${TMP_HOME}/results/logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/all_sequential_$(date +%Y%m%d-%H%M%S).log"

exec > >(tee -a "${LOGFILE}") 2>&1

benchmarks=(humaneval mbpp gsm8k minerva_math_algebra)

echo "========================================="
echo "SEQUENTIAL RUN: 4 models x 4 benchmarks x 2 configs"
echo "Log: ${LOGFILE}"
echo "Started: $(date)"
echo "========================================="

TOTAL_RUNS=32
RUN_NUM=0

run_model() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    local PORT=$3
    local TP=$4
    local GPUS=$5
    local BASE_CONFIG=$6
    local PROWL_CONFIG=$7

    export TP_SIZE="${TP}"
    export CUDA_VISIBLE_DEVICES="${GPUS}"

    echo ""
    echo "========================================="
    echo "MODEL: ${MODEL_NAME} (${MODEL_PATH})"
    echo "  Port: ${PORT}, TP: ${TP}, GPUs: ${GPUS}"
    echo "========================================="

    for config in "${BASE_CONFIG}" "${PROWL_CONFIG}"; do
        local config_name=$(basename "$config" .json)

        for benchmark in ${benchmarks[@]}; do
            RUN_NUM=$((RUN_NUM + 1))
            echo ""
            echo "  [${RUN_NUM}/${TOTAL_RUNS}] ${MODEL_NAME} / ${benchmark} / ${config_name}"
            echo "  $(date)"

            ./run_mixtral_adv_configurable.sh \
                ${PORT} \
                ${MODEL_NAME} \
                ${MODEL_PATH} \
                ${benchmark} \
                "" \
                ${config} \
                ${TP}

            echo "  Done: ${MODEL_NAME} / ${benchmark} / ${config_name}"
        done
    done
}

# DeepSeek-Coder-V2: TP=4, GPUs 0,1,2,3
run_model deepseek_v2 \
    deepseek-ai/DeepSeek-Coder-V2-Instruct \
    8020 4 "0,1,2,3" \
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/config_do_nothing.json" \
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1.125_beta2_optimized.json"

# Mixtral-8x7B: TP=2, GPUs 0,1
run_model mixtral \
    mistralai/Mixtral-8x7B-Instruct-v0.1 \
    8040 2 "0,1" \
    "${TMP_HOME}/prowl/configs/mixtral/do_nothing.json" \
    "${TMP_HOME}/prowl/configs/mixtral/quant_alpha0.7_beta1_optimized.json"

# Qwen2-57B-A14B: TP=2, GPUs 0,1
run_model qwen \
    Qwen/Qwen2-57B-A14B-Instruct \
    8030 2 "0,1" \
    "${TMP_HOME}/prowl/configs/qwen/qwen_do-nothing.json" \
    "${TMP_HOME}/prowl/configs/qwen/quant_alpha3_beta4_optimized.json"

# Qwen3-30B-A3B: TP=1, GPU 0
run_model qwen3 \
    Qwen/Qwen3-30B-A3B-Instruct-2507 \
    8019 1 "0" \
    "${TMP_HOME}/prowl/configs/qwen3_30b/qwen_do-nothing.json" \
    "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha3_beta2_optimized.json"

echo ""
echo "========================================="
echo "ALL SEQUENTIAL RUNS COMPLETED"
echo "Finished: $(date)"
echo "Total runs: ${TOTAL_RUNS}"
echo "========================================="
