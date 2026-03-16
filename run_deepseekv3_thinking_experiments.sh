#!/bin/bash

# Phase-aware expert reduction experiments for thinking models (DeepSeek R1 / V3)
# Tests 4 configurations on AIME benchmark:
#   1. Drop experts in both thinking + decode
#   2. Drop experts only during thinking (not decode)
#   3. Drop experts only during decode (not thinking)
#   4. No dropping (baseline)
#
# IMPORTANT: Verify think_start_token_id and think_end_token_id in configs
# match your model's tokenizer. To check:
#   python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('MODEL_PATH'); print(t.encode('<think>'), t.encode('</think>'))"

export TMP_HOME="${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}"

# --- Configurable parameters ---
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1}"
MODEL_NAME="${MODEL_NAME:-deepseekv3}"
PORT="${PORT:-8020}"
TP_SIZE="${TP_SIZE:-8}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
BENCHMARK="${BENCHMARK:-aime24}"
GPU_WAIT_TIMEOUT="${GPU_WAIT_TIMEOUT:-120}"  # Max seconds to wait for GPU memory

# --- Logging setup ---
LOG_DIR="${TMP_HOME}/results/experiment_logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EXPERIMENT_LOG="${LOG_DIR}/thinking_experiments_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${EXPERIMENT_LOG}"
}

# --- GPU cleanup and readiness check ---
cleanup_gpu() {
    log "Killing any processes on port ${PORT}..."
    lsof -ti:${PORT} | xargs -r kill -9 2>/dev/null
    # Kill any lingering vllm worker processes
    pkill -9 -f "vllm.entrypoints.openai.api_server.*${PORT}" 2>/dev/null
    sleep 5
}

wait_for_gpu_free() {
    local required_free_pct=0.85  # need 85% free to safely start with 0.9 utilization
    local gpu_list="${GPUS//,/ }"
    local waited=0

    log "Waiting for GPUs [${GPUS}] to have sufficient free memory..."
    while [ $waited -lt $GPU_WAIT_TIMEOUT ]; do
        all_free=true
        for gpu_id in $gpu_list; do
            # Get free and total memory in MiB
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

# --- Experiment configs ---
CONFIG_DIR="${TMP_HOME}/prowl/configs/deepseekv3.1"
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
log "Phase-Aware Expert Reduction Experiments"
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
        cleanup_gpu
        wait_for_gpu_free
    fi

    log "Starting experiment: ${label}"
    CUDA_VISIBLE_DEVICES=${GPUS} ./run_mixtral_adv_configurable.sh \
        ${PORT} \
        "${MODEL_NAME}_${label}" \
        "${MODEL_PATH}" \
        "${BENCHMARK}" \
        "" \
        "${config}" \
        ${TP_SIZE} \
        ${MAX_BATCH_SIZE} \
        2>&1 | tee -a "${EXPERIMENT_LOG}"

    exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        log "ERROR: Experiment '${label}' failed with exit code ${exit_code}"
    else
        log "Completed: ${label}"
    fi
    log ""
done

log "========================================="
log "All experiments completed!"
log "Results in: results/"
log "Full log: ${EXPERIMENT_LOG}"
log ""
log "To analyze results:"
log "  python gather_results.py results/"
log "  python analyze_all_benchmark_results.py"
log "========================================="
