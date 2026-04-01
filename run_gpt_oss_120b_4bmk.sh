#!/bin/bash

# GPT-OSS-120B: baseline + alpha3_beta2 on 4 benchmarks

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
MODEL_PATH="openai/gpt-oss-120b"
MODEL_NAME="gpt_oss_120b"
PORT=8050
TP_SIZE=4
GPUS="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

LOGDIR="${TMP_HOME}/results/logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/gpt_oss_120b_4bmk_$(date +%Y%m%d-%H%M%S).log"

# Redirect all output to log file and stdout
exec > >(tee -a "${LOGFILE}") 2>&1

configs=(
    "${TMP_HOME}/prowl/configs/gpt_oss_120b/gpt_oss_do-nothing.json"
    "${TMP_HOME}/prowl/configs/gpt_oss_120b/quant_alpha3_beta2_optimized.json"
)

benchmarks=("humaneval" "mbpp" "gsm8k" "minerva_math_algebra")

echo "Starting GPT-OSS-120B benchmarks on GPUs ${GPUS}, port ${PORT}"
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
echo "GPT-OSS-120B experiment completed! Run gather_results.py to compare:"
echo "  python gather_results.py results/gpt-oss-120b/"
echo "========================================="
