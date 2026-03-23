#!/bin/bash

# Qwen2-57B: baseline + alpha3_beta4 on GPUs 0,1
# 4 benchmarks: humaneval, mbpp, gsm8k, minerva_math_algebra

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
MODEL_PATH="Qwen/Qwen2-57B-A14B-Instruct"
MODEL_NAME="qwen"
PORT=8030
TP_SIZE=2
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

LOGDIR="${TMP_HOME}/results/logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/qwen2_4bmk_$(date +%Y%m%d-%H%M%S).log"

# Redirect all output to log file and stdout
exec > >(tee -a "${LOGFILE}") 2>&1

configs=(
    "${TMP_HOME}/prowl/configs/qwen/qwen_do-nothing.json"
    "${TMP_HOME}/prowl/configs/qwen/quant_alpha3_beta4_optimized.json"
)

benchmarks=("humaneval" "mbpp" "gsm8k" "minerva_math_algebra")

echo "Starting Qwen2-57B benchmarks on GPUs ${GPUS}, port ${PORT}"
echo "Log file: ${LOGFILE}"
echo "Benchmarks: ${benchmarks[*]}"
echo "========================================="

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .json)
    echo ""
    echo "Config: $config_name"
    echo "-----------------------------------------"

    for benchmark in "${benchmarks[@]}"; do
        echo "  Running $benchmark with $config_name"

        ./run_mixtral_adv_configurable.sh \
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
echo "Qwen2 experiment completed! Run gather_results.py to compare:"
echo "  python gather_results.py results/Qwen2-57B-A14B-Instruct/"
echo "========================================="
