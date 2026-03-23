#!/bin/bash

# Qwen3-30B-A3B: baseline + alpha3_beta2 on GPU 4
# 4 benchmarks: humaneval, mbpp, gsm8k, minerva_math_algebra

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_NAME="qwen3"
PORT=8019
TP_SIZE=1
GPUS="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

LOGDIR="${TMP_HOME}/results/logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/qwen3_4bmk_$(date +%Y%m%d-%H%M%S).log"

# Redirect all output to log file and stdout
exec > >(tee -a "${LOGFILE}") 2>&1

configs=(
    "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha3_beta2_optimized.json"
)

benchmarks=("humaneval" "mbpp" "gsm8k" "minerva_math_algebra")

echo "Starting Qwen3-30B benchmarks on GPUs ${GPUS}, port ${PORT}"
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
echo "Qwen3 experiment completed! Run gather_results.py to compare:"
echo "  python gather_results.py results/Qwen3-30B-A3B-Instruct-2507/"
echo "========================================="
