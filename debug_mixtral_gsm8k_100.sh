#!/bin/bash

# Debug script for Mixtral on GSM8K with 100 examples
# Uses only beta 0.7 configuration

echo "========================================="
echo "DEBUG: Mixtral GSM8K - 100 examples"
echo "Beta 0.7 Configuration Only"
echo "========================================="

# Configuration
MODEL_PATH="/scratch/shared_dir/models_dir/Mixtral-8x7B-Instruct-v0.1/"
MODEL_NAME="mixtral"
PORT=8000
TP_SIZE=2
GPUS="0,1"
BENCHMARK="gsm8k"
LIMIT=250  # Only 100 examples for debugging

# Using beta 0.7 configuration
CONFIG_FILE="/nethome/rdudala3/prowl/configs/mixtral/advanced_alpha0_beta0.7.json"

# Extract config name for logging
CONFIG_NAME=$(basename "$CONFIG_FILE" .json)

# The actual STATFILENAME that will be used (set by run_mixtral_adv_configurable.sh)
ACTUAL_STATFILENAME="adv_fp16_${MODEL_NAME}_${BENCHMARK}_port${PORT}"

# Absolute paths for logs and metrics
LOG_DIR="/nethome/rdudala3/latency-scripts/stats/perf/spec_decode/ngram/Mixtral-8x7B-Instruct-v0.1"
METRICS_DIR="/nethome/rdudala3/latency-scripts/stats/quality/${BENCHMARK}/ngram/Mixtral-8x7B-Instruct-v0.1"

# The actual log file names that will be created
PERF_LOG_FILE="${LOG_DIR}/${BENCHMARK}_n${LIMIT}_${ACTUAL_STATFILENAME}_${CONFIG_NAME}.json_port${PORT}.log"
METRICS_FILE="${METRICS_DIR}/${ACTUAL_STATFILENAME}_n${LIMIT}.0_conf_${CONFIG_NAME}.metrics"
EVAL_LOG_FILE="${METRICS_DIR}/${ACTUAL_STATFILENAME}_n${LIMIT}.0_conf_${CONFIG_NAME}.log"

echo "Configuration:"
echo "  Model: ${MODEL_PATH}"
echo "  Port: ${PORT}"
echo "  GPUs: ${GPUS}"
echo "  TP Size: ${TP_SIZE}"
echo "  Benchmark: ${BENCHMARK}"
echo "  Limit: ${LIMIT} examples"
echo "  Config: ${CONFIG_NAME} (Beta 0.7)"
echo "========================================="

# Kill any existing process on the port
echo "Cleaning up port ${PORT}..."
lsof -ti:${PORT} | xargs -r kill -9 2>/dev/null
sleep 2

# Set up environment
export CUDA_VISIBLE_DEVICES=${GPUS}
export TP_SIZE=${TP_SIZE}

# Create directories if they don't exist
mkdir -p ${LOG_DIR}
mkdir -p ${METRICS_DIR}

echo "Starting benchmark run with Beta 0.7..."
echo ""
echo "Expected output files:"
echo "  Performance log: ${PERF_LOG_FILE}"
echo "  Metrics file: ${METRICS_FILE}"
echo "  Evaluation log: ${EVAL_LOG_FILE}"
echo ""

# Run the benchmark with beta 0.7
CUDA_VISIBLE_DEVICES=${GPUS} ./run_mixtral_adv_configurable.sh \
    ${PORT} \
    ${MODEL_NAME} \
    ${MODEL_PATH} \
    ${BENCHMARK} \
    ${LIMIT} \
    ${CONFIG_FILE} \
    ${TP_SIZE}

echo ""
echo "========================================="
echo "Debug run completed with Beta 0.7!"
echo ""
echo "Checking for output files..."
echo ""

# Check if files were created and show their paths
if [ -f "${PERF_LOG_FILE}" ]; then
    echo "✓ Performance log found: ${PERF_LOG_FILE}"
    echo "  Size: $(ls -lh ${PERF_LOG_FILE} | awk '{print $5}')"
else
    echo "✗ Performance log not found at expected location"
    echo "  Checking for similar files..."
    ls -la ${LOG_DIR}/${BENCHMARK}_n${LIMIT}* 2>/dev/null || echo "  No similar files found"
fi

if [ -f "${METRICS_FILE}" ]; then
    echo "✓ Metrics file found: ${METRICS_FILE}"
    echo "  Size: $(ls -lh ${METRICS_FILE} | awk '{print $5}')"
else
    echo "✗ Metrics file not found at expected location"
    echo "  Checking for similar files..."
    ls -la ${METRICS_DIR}/*n${LIMIT}* 2>/dev/null || echo "  No similar files found"
fi

if [ -f "${EVAL_LOG_FILE}" ]; then
    echo "✓ Evaluation log found: ${EVAL_LOG_FILE}"
    echo "  Size: $(ls -lh ${EVAL_LOG_FILE} | awk '{print $5}')"
else
    echo "✗ Evaluation log not found at expected location"
fi

echo ""
echo "To monitor the server log in real-time (if running):"
echo "  tail -f ${PERF_LOG_FILE}"
echo ""
echo "To check throughput and TPOT metrics:"
echo "  grep -E 'Throughput|tpot|TPOT' ${PERF_LOG_FILE}"
echo ""
echo "To check accuracy results:"
echo "  cat ${EVAL_LOG_FILE}"
echo "========================================="
