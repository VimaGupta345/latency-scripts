#!/bin/bash
# Run Qwen3-30B disaggregated at TP=1 (baseline + Prowl) across all benchmarks.
# Requires the GIL-yield fix in gpu_model_runner.py.
#
# Usage:
#   bash run_qwen3_30b_disagg_tp1.sh [prefill_gpu] [decode_gpu] [start_from]
#
# Examples:
#   bash run_qwen3_30b_disagg_tp1.sh 0 1        # all runs
#   bash run_qwen3_30b_disagg_tp1.sh 2 3 5      # resume from run 5

set -e

export TMP_HOME="${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}"
PROWL_ROOT="${TMP_HOME}/prowl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "${PROWL_ROOT}/.venv/bin/activate"

PREFILL_GPU=${1:-0}
DECODE_GPU=${2:-1}
START_FROM=${3:-1}

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
TP_SIZE=1

DO_NOTHING_CONFIG="${PROWL_ROOT}/configs/qwen3_30b/qwen_do-nothing.json"
PROWL_CONFIG="${PROWL_ROOT}/configs/qwen3_30b/quant_alpha1_beta1_optimized.json"

BENCHMARKS=(
    "humaneval|164|humaneval|--trust_remote_code --confirm_run_unsafe_code"
    "mbpp|250|mbpp|--trust_remote_code --confirm_run_unsafe_code --num_fewshot 3"
    "gsm8k|900|gsm8k|--num_fewshot 5"
    "minerva_math_algebra|900|minerva_math_algebra|--num_fewshot 4"
)

CONFIGS=("${DO_NOTHING_CONFIG}" "${PROWL_CONFIG}")
TOTAL_RUNS=$(( ${#CONFIGS[@]} * ${#BENCHMARKS[@]} ))
RUN_NUM=0

echo "========================================="
echo "Qwen3-30B Disagg TP=1 Sweep"
echo "  Model:       ${MODEL}"
echo "  TP:          ${TP_SIZE}"
echo "  Prefill GPU: ${PREFILL_GPU}"
echo "  Decode GPU:  ${DECODE_GPU}"
echo "  Configs:     do-nothing + prowl"
echo "  Benchmarks:  ${#BENCHMARKS[@]}"
echo "  Total runs:  ${TOTAL_RUNS}"
echo "  Started:     $(date)"
echo "========================================="

for config in "${CONFIGS[@]}"; do
    CONFIG_NAME=$(basename "${config}" .json)
    for bench_entry in "${BENCHMARKS[@]}"; do
        IFS='|' read -r BENCH_NAME LIMIT TASKS EXTRA_ARGS <<< "${bench_entry}"
        RUN_NUM=$((RUN_NUM + 1))

        if [ ${RUN_NUM} -lt ${START_FROM} ]; then
            echo "[${RUN_NUM}/${TOTAL_RUNS}] ${CONFIG_NAME} / ${BENCH_NAME} — SKIPPED (before start_from=${START_FROM})"
            continue
        fi

        echo ""
        echo "─────────────────────────────────────"
        echo "[${RUN_NUM}/${TOTAL_RUNS}] ${CONFIG_NAME} / ${BENCH_NAME}"
        echo "$(date)"
        echo "─────────────────────────────────────"

        bash "${SCRIPT_DIR}/run_disagg_lynx.sh" \
            "${MODEL}" \
            "${config}" \
            "${TP_SIZE}" \
            "${PREFILL_GPU}" \
            "${DECODE_GPU}" \
            "${BENCH_NAME}" \
            || echo "  ERROR on run ${RUN_NUM}, continuing..."

        # Extra cleanup: kill any orphan GPU processes between runs
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
            | xargs -r kill -9 2>/dev/null || true
        sleep 5
    done
done

echo ""
echo "========================================="
echo "ALL RUNS COMPLETED"
echo "  Finished: $(date)"
echo "  Total:    ${RUN_NUM}/${TOTAL_RUNS} attempted"
echo "========================================="
