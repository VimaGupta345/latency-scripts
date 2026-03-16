#!/bin/bash

# Sweep max_gen_toks to find how many tokens the thinking model needs for accuracy
# Uses baseline (no_drop) config, 5 prompts, varying max_gen_toks: 2k, 4k, 6k, 8k, 12k, 16k

export TMP_HOME="${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}"

# --- Configurable parameters ---
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
MODEL_NAME="${MODEL_NAME:-qwen3_thinking}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
BENCHMARK="${BENCHMARK:-aime25}"
GPU_WAIT_TIMEOUT="${GPU_WAIT_TIMEOUT:-120}"
CONFIG_FILE="${TMP_HOME}/prowl/configs/qwen3/no_drop.json"
LIMIT=5

# --- max_gen_toks values to sweep ---
GEN_TOKS_VALUES=(2048 4096 6144 8192 12288 16384)

# --- Logging setup ---
LOG_DIR="${TMP_HOME}/results/experiment_logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EXPERIMENT_LOG="${LOG_DIR}/qwen3_gentoks_sweep_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${EXPERIMENT_LOG}"
}

# --- GPU readiness check ---
wait_for_gpu_free() {
    local required_free_pct=0.85
    local gpu_list="${GPUS//,/ }"
    local waited=0

    log "Waiting for GPUs [${GPUS}] to have sufficient free memory..."
    while [ $waited -lt $GPU_WAIT_TIMEOUT ]; do
        all_free=true
        for gpu_id in $gpu_list; do
            read free_mib total_mib <<< $(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits -i $gpu_id | tr ',' ' ')
            free_pct=$(awk "BEGIN {printf \"%.2f\", $free_mib / $total_mib}")
            if (( $(awk "BEGIN {print ($free_pct < $required_free_pct)}") )); then
                all_free=false
                log "  GPU $gpu_id: ${free_mib}/${total_mib} MiB free (${free_pct}) - waiting..."
                break
            fi
        done
        if $all_free; then
            log "All GPUs ready."
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    log "WARNING: GPUs not fully free after ${GPU_WAIT_TIMEOUT}s, proceeding anyway..."
    return 1
}

# --- Start vLLM server ---
start_server() {
    local modelname=$(basename ${MODEL_PATH})
    local configname=$(basename ${CONFIG_FILE})
    local stats_dir="${TMP_HOME}/results/server_logs/${modelname}"
    mkdir -p "${stats_dir}"
    local stat_file="gentoks_sweep_${configname}_port${PORT}.log"

    log "Starting vLLM server: model=${MODEL_PATH}, config=${CONFIG_FILE}, port=${PORT}"

    source "${TMP_HOME}/prowl/.venv/bin/activate"

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
        2>&1 | tee ${stats_dir}/${stat_file} &

    SERVER_PID=$!
    log "Server PID: ${SERVER_PID}"

    local max_wait=900
    local elapsed=0
    log "Waiting for server at http://localhost:${PORT}/health (timeout=${max_wait}s)..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null | grep -q 200; then
            log "Server is ready after ${elapsed}s"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    log "ERROR: Server not ready after ${max_wait}s"
    return 1
}

# --- Run lm-eval with specific max_gen_toks ---
run_lm_eval() {
    local gen_toks=$1

    local modelname=$(basename ${MODEL_PATH})
    local stats_dir="${TMP_HOME}/results/${modelname}/${BENCHMARK}"
    mkdir -p "${stats_dir}"

    local stat_file="gentoks_sweep_${MODEL_NAME}_${BENCHMARK}_gentoks${gen_toks}_n${LIMIT}"

    local lm_eval_cmd="lm-eval --model local-chat-completions \
        --tasks ${BENCHMARK} \
        --apply_chat_template \
        --model_args model=${MODEL_PATH},base_url=http://localhost:${PORT}/v1/chat/completions,num_concurrent=${LIMIT},timeout=1800 \
        --gen_kwargs max_gen_toks=${gen_toks} \
        --limit ${LIMIT} --log_samples --num_fewshot 0 --trust_remote_code \
        --output_path ${stats_dir}/${stat_file} \
        2>&1 | tee ${stats_dir}/${stat_file}.log"

    log "eval_cmd: ${lm_eval_cmd}"
    eval ${lm_eval_cmd}
    return $?
}

# --- Kill server ---
kill_server() {
    if [ -n "${SERVER_PID}" ]; then
        log "Stopping server (PID: ${SERVER_PID}) and all child processes..."
        kill -9 -$(ps -o pgid= -p ${SERVER_PID} 2>/dev/null | tr -d ' ') 2>/dev/null
        wait ${SERVER_PID} 2>/dev/null
    fi
    lsof -ti:${PORT} | xargs -r kill -9 2>/dev/null
    sleep 3
}

# --- Main ---
log "========================================="
log "Qwen3 max_gen_toks Sweep"
log "========================================="
log "Model:     ${MODEL_PATH}"
log "Config:    ${CONFIG_FILE}"
log "Benchmark: ${BENCHMARK}"
log "Limit:     ${LIMIT} prompts"
log "TP Size:   ${TP_SIZE}"
log "GPUs:      ${GPUS}"
log "Port:      ${PORT}"
log "Gen toks:  ${GEN_TOKS_VALUES[*]}"
log "Log file:  ${EXPERIMENT_LOG}"
log "========================================="

# Start server once — reuse for all gen_toks values
start_server
if [ $? -ne 0 ]; then
    log "ERROR: Server failed to start"
    kill_server
    exit 1
fi

for i in "${!GEN_TOKS_VALUES[@]}"; do
    gen_toks="${GEN_TOKS_VALUES[$i]}"

    log ""
    log "-----------------------------------------"
    log "Run $((i+1))/${#GEN_TOKS_VALUES[@]}: max_gen_toks=${gen_toks}"
    log "-----------------------------------------"

    run_lm_eval "${gen_toks}"
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        log "ERROR: Run max_gen_toks=${gen_toks} failed with exit code ${exit_code}"
    else
        log "Completed: max_gen_toks=${gen_toks}"
    fi
    log ""
done

kill_server

log "========================================="
log "All runs completed!"
log "Results in: ${TMP_HOME}/results/$(basename ${MODEL_PATH})/${BENCHMARK}/"
log "Full log: ${EXPERIMENT_LOG}"
log "========================================="
