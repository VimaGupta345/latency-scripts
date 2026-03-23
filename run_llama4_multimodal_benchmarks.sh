#!/usr/bin/env zsh

# Run Llama 4 Scout multimodal benchmarks SEQUENTIALLY
# Usage: CUDA_VISIBLE_DEVICES=4,5 zsh run_llama4_multimodal_benchmarks.sh

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source

# Activate prowl venv
source "${TMP_HOME}/prowl/.venv/bin/activate"

LOGDIR="${TMP_HOME}/results/logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/llama4_multimodal_$(date +%Y%m%d-%H%M%S).log"

exec > >(tee -a "${LOGFILE}") 2>&1

benchmarks=(chartqa mmmu_val)

echo "========================================="
echo "SEQUENTIAL RUN: 1 model x 2 benchmarks x 2 configs"
echo "Log: ${LOGFILE}"
echo "Started: $(date)"
echo "========================================="

TOTAL_RUNS=4
RUN_NUM=0

run_model() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    local TP=$3
    local BASE_CONFIG=$4
    local PROWL_CONFIG=$5

    echo ""
    echo "========================================="
    echo "MODEL: ${MODEL_NAME} (${MODEL_PATH})"
    echo "  TP: ${TP}"
    echo "========================================="

    for config in "${BASE_CONFIG}" "${PROWL_CONFIG}"; do
        local config_name=$(basename "$config" .json)

        for benchmark in ${benchmarks[@]}; do
            RUN_NUM=$((RUN_NUM + 1))
            echo ""
            echo "  [${RUN_NUM}/${TOTAL_RUNS}] ${MODEL_NAME} / ${benchmark} / ${config_name}"
            echo "  $(date)"

            python lm_eval_multimodal_serve.py \
                -m ${MODEL_PATH} \
                -o "adv_fp16_${MODEL_NAME}_${benchmark}" \
                -b ${benchmark} \
                -cf ${config} \
                -tp ${TP}

            echo "  Done: ${MODEL_NAME} / ${benchmark} / ${config_name}"
        done
    done
}

# Llama 4 Scout-17B-16E: TP=2
run_model llama4_scout \
    meta-llama/Llama-4-Scout-17B-16E-Instruct \
    2 \
    "${TMP_HOME}/prowl/configs/llama4/do-nothing.json" \
    "${TMP_HOME}/prowl/configs/llama4/quant_alpha3.json"

echo ""
echo "========================================="
echo "ALL SEQUENTIAL RUNS COMPLETED"
echo "Finished: $(date)"
echo "Total runs: ${TOTAL_RUNS}"
echo "========================================="
