#!/bin/bash
set -euo pipefail

# Llama4 multimodal evals (ChartQA + MMMU) with "do-nothing" routing.
#
# Override any of these via env:
#   MODEL_PATH, MODEL_NAME, PORT, TP_SIZE, GPUS, MAX_BATCH_SIZE,
#   CONFIG_FILE, ENABLE_EXPERT_PARALLEL, CHARTQA_TASK, MMMU_TASK, TMP_HOME

MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-4-Scout-17B-16E-Instruct"}
MODEL_NAME=${MODEL_NAME:-"llama4"}
PORT=${PORT:-8011}
TP_SIZE=${TP_SIZE:-4}
GPUS=${GPUS:-"0,1,2,3"}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-16}
TMP_HOME=${TMP_HOME:-/nethome/rdudala3}

CONFIG_FILE=${CONFIG_FILE:-"${TMP_HOME}/prowl/configs/llama4/do-nothing.json"}
ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL:-false}

CHARTQA_TASK=${CHARTQA_TASK:-chartqa}
MMMU_TASK=${MMMU_TASK:-mmmu_val}

benchmarks=("${CHARTQA_TASK}" "${MMMU_TASK}")

echo "Model: ${MODEL_PATH}"
echo "Config: ${CONFIG_FILE}"
echo "Benchmarks: ${benchmarks[*]}"
echo "GPUs: ${GPUS} | TP: ${TP_SIZE} | Max batch: ${MAX_BATCH_SIZE} | Port: ${PORT}"
echo "Expert parallel: ${ENABLE_EXPERT_PARALLEL}"

for benchmark in "${benchmarks[@]}"; do
  echo "Running ${benchmark}..."
  TMP_HOME=${TMP_HOME} CUDA_VISIBLE_DEVICES=${GPUS} \
    ./run_mixtral_adv_configurable.sh \
      ${PORT} \
      ${MODEL_NAME} \
      ${MODEL_PATH} \
      ${benchmark} \
      "" \
      ${CONFIG_FILE} \
      ${TP_SIZE} \
      ${MAX_BATCH_SIZE} \
      ${ENABLE_EXPERT_PARALLEL}
done
