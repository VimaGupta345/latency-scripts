#!/bin/bash

# benchmark_qwen2_gsm8k.sh - Measure TPOT/throughput of Qwen2-57B-A14B-Instruct on GSM8K
#
# Toggles the vLLM attention backend via VLLM_ATTENTION_BACKEND env var.
# See: https://docs.vllm.ai/en/stable/configuration/env_vars/
#
# Usage:
#   ./benchmark_qwen2_gsm8k.sh baseline                # FLASH_ATTN (default)
#   ./benchmark_qwen2_gsm8k.sh flashinfer FLASHINFER   # FlashInfer backend, auto-compares to baseline

set -euo pipefail

LABEL=${1:?"Usage: $0 <label> [attention_backend]  (e.g., baseline FLASH_ATTN, flashinfer FLASHINFER)"}
ATTENTION_BACKEND=${2:-FLASH_ATTN}

# --- Configuration ---
MODEL="Qwen/Qwen2-57B-A14B-Instruct"
MODEL_NAME="Qwen2-57B-A14B-Instruct"
BENCHMARK="gsm8k"
LIMIT=200
PORT=${PORT:-8005}
TP_SIZE=${TP_SIZE:-2}
GPUS=${CUDA_VISIBLE_DEVICES:-"5,6"}
SLEEP_TIME=${SLEEP_TIME:-120}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-16}
SERVER_ADDRESS="localhost:${PORT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
STATS_BASE_DIR="${SCRIPT_DIR}/stats"
CONFIG_FILE="${SCRIPT_DIR}/../prowl/configs/qwen/qwen_do-nothing.json"
METRICS_DIR="${STATS_BASE_DIR}/quality/${BENCHMARK}/ngram/${MODEL_NAME}"

mkdir -p "${RESULTS_DIR}"

STATFILENAME="${LABEL}_qwen2_gsm8k"

echo "==========================================="
echo "  Qwen2 GSM8K Attention Backend Benchmark"
echo "==========================================="
echo "  Label:             ${LABEL}"
echo "  Attention Backend: ${ATTENTION_BACKEND}"
echo "  Model:             ${MODEL}"
echo "  Benchmark:         ${BENCHMARK}"
echo "  Limit:             ${LIMIT} requests"
echo "  Port:              ${PORT}"
echo "  TP Size:           ${TP_SIZE}"
echo "  GPUs:              ${GPUS}"
echo "  Config:            ${CONFIG_FILE}"
echo "  Sleep Time:        ${SLEEP_TIME}s"
echo "==========================================="

# --- Clean up port before starting ---
echo "Cleaning up port ${PORT}..."
lsof -ti:"${PORT}" | xargs -r kill -9 2>/dev/null || true
sleep 2

# --- Run the benchmark via lm_eval_online_serve.py ---
echo "Starting benchmark run: ${LABEL} (backend=${ATTENTION_BACKEND})"
export TP_SIZE="${TP_SIZE}"
export VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND}"
export STATS_BASE_DIR="${STATS_BASE_DIR}"

CUDA_VISIBLE_DEVICES=${GPUS} python "${SCRIPT_DIR}/lm_eval_online_serve.py" \
    -m "${MODEL}" \
    -o "${STATFILENAME}" \
    -b "${BENCHMARK}" \
    -l "${LIMIT}" \
    -k 0 \
    -t "${SLEEP_TIME}" \
    -cf "${CONFIG_FILE}" \
    -sa "${SERVER_ADDRESS}" \
    -mb "${MAX_BATCH_SIZE}" \
    -s "${SCRIPT_DIR}/online_serving_ngram_port.sh"

echo "Benchmark run complete."

# --- Find the latest .metrics file for this run ---
METRICS_PATTERN="${STATFILENAME}_n${LIMIT}_conf_qwen_do-nothing_*.metrics"
METRICS_FILE=$(ls -t "${METRICS_DIR}"/${METRICS_PATTERN} 2>/dev/null | head -1)

if [ -z "${METRICS_FILE}" ]; then
    echo "ERROR: No metrics file found matching ${METRICS_DIR}/${METRICS_PATTERN}"
    exit 1
fi

echo "Found metrics file: ${METRICS_FILE}"

# --- Parse metrics and save summary ---
echo ""
echo "==========================================="
echo "  Metrics Summary for: ${LABEL}"
echo "==========================================="
METRICS_OUTPUT=$(python "${SCRIPT_DIR}/get_vllm_metrics.py" "${METRICS_FILE}")
echo "${METRICS_OUTPUT}"

# Extract TPOT and throughput values
TPOT=$(echo "${METRICS_OUTPUT}" | grep -oP 'TPOT: \K[0-9.]+')
GEN_THROUGHPUT=$(echo "${METRICS_OUTPUT}" | grep -oP 'Generation Throughput: \K[0-9.]+')
OVERALL_THROUGHPUT=$(echo "${METRICS_OUTPUT}" | grep -oP 'Overall Throughput:\s+\K[0-9.]+')

# Save summary
SUMMARY_FILE="${RESULTS_DIR}/${LABEL}.txt"
cat > "${SUMMARY_FILE}" <<EOF
label=${LABEL}
attention_backend=${ATTENTION_BACKEND}
tpot_ms=${TPOT}
gen_throughput_tps=${GEN_THROUGHPUT}
overall_throughput_tps=${OVERALL_THROUGHPUT}
metrics_file=${METRICS_FILE}
timestamp=$(date -Iseconds)
EOF

echo ""
echo "Summary saved to: ${SUMMARY_FILE}"

# --- Delta comparison against baseline ---
BASELINE_FILE="${RESULTS_DIR}/baseline.txt"

if [ "${LABEL}" != "baseline" ] && [ -f "${BASELINE_FILE}" ]; then
    BASELINE_TPOT=$(grep -oP 'tpot_ms=\K[0-9.]+' "${BASELINE_FILE}")
    BASELINE_GEN=$(grep -oP 'gen_throughput_tps=\K[0-9.]+' "${BASELINE_FILE}")
    BASELINE_OVERALL=$(grep -oP 'overall_throughput_tps=\K[0-9.]+' "${BASELINE_FILE}")

    echo ""
    echo "==========================================="
    echo "  Delta: ${LABEL} vs baseline"
    echo "==========================================="

    TPOT_RATIO=$(awk "BEGIN {printf \"%.3f\", ${BASELINE_TPOT} / ${TPOT}}")
    GEN_RATIO=$(awk "BEGIN {printf \"%.3f\", ${GEN_THROUGHPUT} / ${BASELINE_GEN}}")
    OVERALL_RATIO=$(awk "BEGIN {printf \"%.3f\", ${OVERALL_THROUGHPUT} / ${BASELINE_OVERALL}}")

    TPOT_DELTA=$(awk "BEGIN {printf \"%.2f\", ${TPOT} - ${BASELINE_TPOT}}")
    GEN_DELTA=$(awk "BEGIN {printf \"%.2f\", ${GEN_THROUGHPUT} - ${BASELINE_GEN}}")
    OVERALL_DELTA=$(awk "BEGIN {printf \"%.2f\", ${OVERALL_THROUGHPUT} - ${BASELINE_OVERALL}}")

    printf "  %-25s %12s %12s %12s %10s\n" "Metric" "baseline" "${LABEL}" "delta" "speedup"
    printf "  %-25s %12s %12s %12s %10s\n" "-------------------------" "------------" "------------" "------------" "----------"
    printf "  %-25s %10s ms %10s ms %10s ms %9sx\n" "TPOT" "${BASELINE_TPOT}" "${TPOT}" "${TPOT_DELTA}" "${TPOT_RATIO}"
    printf "  %-25s %9s t/s %9s t/s %9s t/s %9sx\n" "Gen Throughput" "${BASELINE_GEN}" "${GEN_THROUGHPUT}" "${GEN_DELTA}" "${GEN_RATIO}"
    printf "  %-25s %9s t/s %9s t/s %9s t/s %9sx\n" "Overall Throughput" "${BASELINE_OVERALL}" "${OVERALL_THROUGHPUT}" "${OVERALL_DELTA}" "${OVERALL_RATIO}"
    echo "==========================================="
elif [ "${LABEL}" = "baseline" ]; then
    echo "Baseline recorded. Run again with a different label to compare."
else
    echo "No baseline found at ${BASELINE_FILE}. Run with 'baseline' label first to enable comparison."
fi

echo "Done."
