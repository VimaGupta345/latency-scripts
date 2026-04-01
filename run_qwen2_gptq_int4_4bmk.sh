#!/bin/bash

# Qwen2-57B GPTQ-Int4: baseline + alpha3_beta4 on a single GPU
# 4 benchmarks: humaneval, mbpp, gsm8k, minerva_math_algebra
# Compare accuracy drop and perf benefit vs FP16 variant

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
MODEL_PATH="Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4"
MODEL_NAME="qwen"
PORT=8031
TP_SIZE=1  # Int4 model fits on 1 GPU (~30GB)
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

LOGDIR="${TMP_HOME}/results/logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/qwen2_gptq_int4_4bmk_$(date +%Y%m%d-%H%M%S).log"

# Redirect all output to log file and stdout
exec > >(tee -a "${LOGFILE}") 2>&1

configs=(
    "${TMP_HOME}/prowl/configs/qwen/qwen_do-nothing.json"
    "${TMP_HOME}/prowl/configs/qwen/quant_alpha3_beta4_optimized.json"
)

benchmarks=("humaneval" "mbpp" "gsm8k" "minerva_math_algebra")

echo "Starting Qwen2-57B GPTQ-Int4 benchmarks on GPU ${GPUS}, port ${PORT}"
echo "Log file: ${LOGFILE}"
echo "Benchmarks: ${benchmarks[*]}"
echo "TP_SIZE: ${TP_SIZE} (Int4 model fits on 1 GPU)"
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
echo "Qwen2 GPTQ-Int4 experiment completed! Run gather_results.py to compare:"
echo "  python gather_results.py results/Qwen2-57B-A14B-Instruct-GPTQ-Int4/"
echo ""
echo "To compare vs FP16 baseline:"
echo "  python gather_results.py results/Qwen2-57B-A14B-Instruct/"
echo "========================================="
