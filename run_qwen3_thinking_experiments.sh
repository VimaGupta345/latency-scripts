#!/bin/bash

# Phase-aware expert reduction experiments for Qwen3 thinking models
# Tests 4 configurations on AIME25 benchmark:
#   1. Drop experts in both thinking + decode
#   2. Drop experts only during thinking (not decode)
#   3. Drop experts only during decode (not thinking)
#   4. No dropping (baseline)
#
# Qwen3-30B-A3B-Thinking-2507 think tokens:
#   <think> = 151667, </think> = 151668
# Verified via: python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('Qwen/Qwen3-30B-A3B-Thinking-2507'); print(t.encode('<think>'), t.encode('</think>'))"

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

# --- Logging setup ---
LOG_DIR="${TMP_HOME}/results/experiment_logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EXPERIMENT_LOG="${LOG_DIR}/qwen3_thinking_experiments_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${EXPERIMENT_LOG}"
}

# --- GPU cleanup and readiness check ---
cleanup_gpu() {
    log "Killing any processes on port ${PORT}..."
    lsof -ti:${PORT} | xargs -r kill -9 2>/dev/null
    pkill -9 -f "vllm.entrypoints.openai.api_server.*${PORT}" 2>/dev/null
    sleep 5
}

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

# --- Start vLLM server with reasoning parser ---
start_server() {
    local config_file=$1
    local statfilename=$2

    local modelname=$(basename ${MODEL_PATH})
    local configname=$(basename ${config_file})
    local stats_dir="${TMP_HOME}/results/server_logs/${modelname}"
    mkdir -p "${stats_dir}"
    local stat_file="${statfilename}_${configname}_port${PORT}.log"

    log "Starting vLLM server: model=${MODEL_PATH}, config=${config_file}, port=${PORT}"

    source "${TMP_HOME}/prowl/.venv/bin/activate"

    setsid vllm serve ${MODEL_PATH} \
        --host localhost \
        --port ${PORT} \
        --max-num-seqs ${MAX_BATCH_SIZE} \
        --tensor-parallel-size ${TP_SIZE} \
        --enforce-eager \
        --gpu-memory-utilization 0.9 \
        --mixtral_config_file ${config_file} \
        --reasoning-parser qwen3 \
        --trust-remote-code \
        2>&1 | tee ${stats_dir}/${stat_file} &

    SERVER_PID=$!
    log "Server PID: ${SERVER_PID}"

    # Wait for server health
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

# --- Run lm-eval benchmark ---
run_lm_eval() {
    local config_file=$1
    local label=$2

    local modelname=$(basename ${MODEL_PATH})
    local conf_name=$(basename "${config_file}" .json)
    local stats_dir="${TMP_HOME}/results/${modelname}/${BENCHMARK}"
    mkdir -p "${stats_dir}"

    local stat_file="adv_fp16_${MODEL_NAME}_${label}_${BENCHMARK}_port${PORT}_n30_conf_${conf_name}"

    # Query max_model_len from server
    local max_len=$(curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0].get('max_model_len', 32768))" 2>/dev/null || echo 32768)

    local lm_eval_cmd="lm-eval --model local-chat-completions \
        --tasks ${BENCHMARK} \
        --apply_chat_template \
        --model_args model=${MODEL_PATH},base_url=http://localhost:${PORT}/v1/chat/completions,num_concurrent=20,timeout=1800 \
        --gen_kwargs max_gen_toks=8192 \
        --limit 30 --log_samples --num_fewshot 0 --trust_remote_code \
        --output_path ${stats_dir}/${stat_file} \
        2>&1 | tee ${stats_dir}/${stat_file}.log"

    log "eval_cmd: ${lm_eval_cmd}"
    eval ${lm_eval_cmd}
    local exit_code=$?

    # Collect server metrics
    log "Collecting metrics for ${stat_file}"
    local timestamp=$(date +%Y%m%d-%H%M%S)
    curl -s "http://localhost:${PORT}/metrics" > "${stats_dir}/${stat_file}_${timestamp}.metrics" 2>/dev/null

    return ${exit_code}
}

# --- Kill server ---
kill_server() {
    if [ -n "${SERVER_PID}" ]; then
        log "Stopping server (PID: ${SERVER_PID}) and all child processes..."
        # Kill entire process group (setsid ensures vllm and all workers share a group)
        kill -9 -$(ps -o pgid= -p ${SERVER_PID} 2>/dev/null | tr -d ' ') 2>/dev/null
        wait ${SERVER_PID} 2>/dev/null
    fi
    # Also kill anything on the port
    lsof -ti:${PORT} | xargs -r kill -9 2>/dev/null
    sleep 3
}

# --- Experiment configs ---
CONFIG_DIR="${TMP_HOME}/prowl/configs/qwen3"
configs=(
    "${CONFIG_DIR}/quant_drop_both.json"
    "${CONFIG_DIR}/quant_drop_thinking_only.json"
    "${CONFIG_DIR}/quant_drop_decode_only.json"
    "${CONFIG_DIR}/no_drop.json"
)
config_labels=(
    "drop_both"
    "drop_thinking_only"
    "drop_decode_only"
    "no_drop_baseline"
)

log "========================================="
log "Qwen3 Phase-Aware Expert Reduction Experiments"
log "========================================="
log "Model:     ${MODEL_PATH}"
log "Benchmark: ${BENCHMARK}"
log "TP Size:   ${TP_SIZE}"
log "GPUs:      ${GPUS}"
log "Port:      ${PORT}"
log "Log file:  ${EXPERIMENT_LOG}"
log "========================================="

for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    label="${config_labels[$i]}"
    config_name=$(basename "$config" .json)

    log ""
    log "-----------------------------------------"
    log "Experiment $((i+1))/4: ${label}"
    log "Config: ${config_name}"
    log "-----------------------------------------"

    # Clean up previous server and wait for GPUs
    if [ $i -gt 0 ]; then
        kill_server
        wait_for_gpu_free
    fi

    # Start server with this config
    start_server "${config}" "${BENCHMARK}_ndef_adv_fp16_${MODEL_NAME}_${label}_${BENCHMARK}_port${PORT}"

    if [ $? -ne 0 ]; then
        log "ERROR: Server failed to start for experiment '${label}'"
        kill_server
        continue
    fi

    # Run lm-eval
    run_lm_eval "${config}" "${label}"
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        log "ERROR: Experiment '${label}' failed with exit code ${exit_code}"
    else
        log "Completed: ${label}"
    fi

    # Kill server before next experiment
    kill_server
    log ""
done

log "========================================="
log "All experiments completed!"
log "Results in: ${TMP_HOME}/results/Qwen3-30B-A3B-Thinking-2507/${BENCHMARK}/"
log "Full log: ${EXPERIMENT_LOG}"
log ""
log "To analyze results:"
log "  python gather_results.py results/"
log "  python analyze_all_benchmark_results.py"
log "========================================="
