#!/bin/bash

# Quick baseline run with batch type logging enabled

export TMP_HOME="${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
MODEL_NAME="${MODEL_NAME:-qwen3_thinking}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
BENCHMARK="${BENCHMARK:-aime25}"
CONFIG_FILE="${TMP_HOME}/prowl/configs/qwen3/no_drop.json"

LOG_DIR="${TMP_HOME}/results/experiment_logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EXPERIMENT_LOG="${LOG_DIR}/qwen3_baseline_debug_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${EXPERIMENT_LOG}"
}

kill_server() {
    if [ -n "${SERVER_PID}" ]; then
        log "Stopping server (PID: ${SERVER_PID}) and all child processes..."
        kill -9 -$(ps -o pgid= -p ${SERVER_PID} 2>/dev/null | tr -d ' ') 2>/dev/null
        wait ${SERVER_PID} 2>/dev/null
    fi
    lsof -ti:${PORT} | xargs -r kill -9 2>/dev/null
    sleep 3
}

trap kill_server EXIT

log "========================================="
log "Qwen3 Baseline Debug (batch type logging)"
log "========================================="
log "Model:     ${MODEL_PATH}"
log "Config:    ${CONFIG_FILE}"
log "Benchmark: ${BENCHMARK}"
log "TP Size:   ${TP_SIZE}"
log "GPUs:      ${GPUS}"
log "Port:      ${PORT}"
log "========================================="

source "${TMP_HOME}/prowl/.venv/bin/activate"

modelname=$(basename ${MODEL_PATH})
stats_dir="${TMP_HOME}/results/server_logs/${modelname}"
mkdir -p "${stats_dir}"
SERVER_LOG="${stats_dir}/baseline_debug_${TIMESTAMP}_port${PORT}.log"

log "Starting vLLM server..."
setsid vllm serve ${MODEL_PATH} \
    --host localhost \
    --port ${PORT} \
    --max-num-seqs ${MAX_BATCH_SIZE} \
    --tensor-parallel-size ${TP_SIZE} \
    --enforce-eager \
    --gpu-memory-utilization 0.9 \
    --mixtral_config_file ${CONFIG_FILE} \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    2>&1 | tee ${SERVER_LOG} &

SERVER_PID=$!
log "Server PID: ${SERVER_PID}"

max_wait=900
elapsed=0
log "Waiting for server health..."
while [ $elapsed -lt $max_wait ]; do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null | grep -q 200; then
        log "Server ready after ${elapsed}s"
        break
    fi
    sleep 10
    elapsed=$((elapsed + 10))
done

if [ $elapsed -ge $max_wait ]; then
    log "ERROR: Server not ready"
    exit 1
fi

results_dir="${TMP_HOME}/results/${modelname}/${BENCHMARK}"
mkdir -p "${results_dir}"
stat_file="baseline_debug_${MODEL_NAME}_${BENCHMARK}_gentoks8192_n30"

log "Running lm-eval..."
lm-eval --model local-chat-completions \
    --tasks ${BENCHMARK} \
    --apply_chat_template \
    --model_args model=${MODEL_PATH},base_url=http://localhost:${PORT}/v1/chat/completions,num_concurrent=20,timeout=1800 \
    --gen_kwargs max_gen_toks=8192 \
    --limit 30 --log_samples --num_fewshot 0 --trust_remote_code \
    --output_path ${results_dir}/${stat_file} \
    2>&1 | tee ${results_dir}/${stat_file}.log

log "Eval done. Collecting metrics..."
curl -s "http://localhost:${PORT}/metrics" > "${results_dir}/${stat_file}_${TIMESTAMP}.metrics" 2>/dev/null

log "Stopping server to trigger batch type summary dump..."
kill_server

log "========================================="
log "Done! Check server log for [PROWL BATCH STATS]:"
log "  ${SERVER_LOG}"
log "========================================="
