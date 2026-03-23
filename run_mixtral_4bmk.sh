#!/bin/bash

# Mixtral-8x7B: baseline + alpha0.7_beta1 on GPUs 2,3
# 4 benchmarks: humaneval, mbpp, gsm8k, minerva_math_algebra

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
MODEL_PATH="mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_NAME="mixtral"
PORT=8040
TP_SIZE=2
GPUS="${CUDA_VISIBLE_DEVICES:-2,3}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

LOGDIR="${TMP_HOME}/results/logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/mixtral_4bmk_$(date +%Y%m%d-%H%M%S).log"

# Redirect all output to log file and stdout
exec > >(tee -a "${LOGFILE}") 2>&1

configs=(
    "${TMP_HOME}/prowl/configs/mixtral/do_nothing.json"
    "${TMP_HOME}/prowl/configs/mixtral/quant_alpha0.7_beta1_optimized.json"
)

benchmarks=("humaneval" "mbpp" "gsm8k" "minerva_math_algebra")

echo "Starting Mixtral-8x7B benchmarks on GPUs ${GPUS}, port ${PORT}"
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
echo "Mixtral experiment completed! Run gather_results.py to compare:"
echo "  python gather_results.py results/Mixtral-8x7B-Instruct-v0.1/"
echo "========================================="
