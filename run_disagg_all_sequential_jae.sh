#!/bin/bash
# Run all models x benchmarks x configs SEQUENTIALLY in disaggregated mode.
#
# Each run launches prefill + decode + proxy, runs the benchmark, collects
# metrics, then tears everything down before the next run.
#
# Usage:
#   bash run_disagg_all_sequential.sh [prefill_gpus] [decode_gpus] [start_from]
#
# Examples:
#   bash run_disagg_all_sequential.sh 0,1 2,3          # all runs from the start
#   bash run_disagg_all_sequential.sh 0,1,2,3 4,5,6,7  # TP=4 models on 8 GPUs
#   bash run_disagg_all_sequential.sh 0,1,2,3 4,5,6,7 13  # resume from run 13

set -e

export TMP_HOME="${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}"
PROWL_ROOT="${TMP_HOME}/prowl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "${PROWL_ROOT}/.venv/bin/activate"

PREFILL_GPUS=${1:-0,1}
DECODE_GPUS=${2:-2,3}
START_FROM=${3:-1}  # Skip runs before this number (e.g. 13 to resume from run 13)

# --- Ports (fixed across all runs) ---
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8000

# --- Log setup ---
LOGDIR="${TMP_HOME}/results/logs/disagg_sweep"
mkdir -p "${LOGDIR}"
MASTER_LOG="${LOGDIR}/sweep_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

# --- Benchmark definitions with safe limits (<=900 per memory note) ---
# Format: "benchmark_name|limit|tasks|extra_args"
BENCHMARKS=(
    "humaneval|164|humaneval|--trust_remote_code --confirm_run_unsafe_code"
    "mbpp|250|mbpp|--trust_remote_code --confirm_run_unsafe_code --num_fewshot 3"
    "gsm8k|900|gsm8k|--num_fewshot 5"
    "minerva_math_algebra|900|minerva_math_algebra|--num_fewshot 4"
)

# --- Model definitions ---
# Format: "name|path|tp_size|do_nothing_config|prowl_config"
MODELS=(
    "qwen|Qwen/Qwen2-57B-A14B-Instruct|2|${PROWL_ROOT}/configs/qwen/qwen_do-nothing.json|${PROWL_ROOT}/configs/qwen/quant_alpha3_beta4_optimized.json"
    "qwen|Qwen/Qwen2-57B-A14B-Instruct|2|${PROWL_ROOT}/configs/qwen/qwen_do-nothing.json|${PROWL_ROOT}/configs/qwen/quant_alpha3_beta6_optimized.json"
    "qwen|Qwen/Qwen2-57B-A14B-Instruct|2|${PROWL_ROOT}/configs/qwen/qwen_do-nothing.json|${PROWL_ROOT}/configs/qwen/quant_alpha4_beta5_optimized.json"
    "qwen|Qwen/Qwen2-57B-A14B-Instruct|2|${PROWL_ROOT}/configs/qwen/qwen_do-nothing.json|${PROWL_ROOT}/configs/qwen/quant_alpha4_beta7_optimized.json"
    "qwen3|Qwen/Qwen3-30B-A3B-Instruct-2507|2|${PROWL_ROOT}/configs/qwen3_30b/qwen_do-nothing.json|${PROWL_ROOT}/configs/qwen3_30b/quant_alpha1_beta1_optimized.json"
    "mixtral|mistralai/Mixtral-8x7B-Instruct-v0.1|2|${PROWL_ROOT}/configs/mixtral/do_nothing.json|${PROWL_ROOT}/configs/mixtral/quant_alpha0.7_beta1_optimized.json"
    "deepseek_v2|deepseek-ai/DeepSeek-Coder-V2-Instruct|4|${PROWL_ROOT}/configs/deepseek_v2_coder/config_do_nothing.json|${PROWL_ROOT}/configs/deepseek_v2_coder/quant_alpha1.125_beta2_optimized.json"
)

# --- Tunable parameters ---
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.9
MAX_BATCH_SIZE=16
PREFILL_SIDE_CHANNEL=5559
DECODE_SIDE_CHANNEL=5659

export VLLM_USE_V1=1

# =========================================================================
# Helper: force-kill anything on our ports
# =========================================================================
kill_ports() {
    # Kill processes on the ports AND their entire session trees.
    # vLLM spawns TP worker children via multiprocessing that don't listen
    # on the port — killing only the port holder leaves orphan workers
    # holding GPU memory. Since servers are launched via setsid, we kill
    # the session (SID) to catch all descendants.
    for port in ${PREFILL_PORT} ${DECODE_PORT} ${PROXY_PORT}; do
        for pid in $(lsof -ti:${port} 2>/dev/null); do
            local sid=$(ps -o sid= -p ${pid} 2>/dev/null | tr -d ' ')
            if [ -n "${sid}" ] && [ "${sid}" != "0" ]; then
                kill -9 -${sid} 2>/dev/null || true
            fi
            kill -9 ${pid} 2>/dev/null || true
        done
    done
    # Also kill any orphan GPU processes owned by us that survived the
    # session kill (TP workers that detached from their parent).
    local my_uid=$(id -u)
    for gpu_pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        gpu_pid=$(echo "$gpu_pid" | tr -d ' ')
        local owner=$(ps -o uid= -p "$gpu_pid" 2>/dev/null | tr -d ' ')
        if [ "$owner" = "$my_uid" ]; then
            kill -9 "$gpu_pid" 2>/dev/null || true
        fi
    done
    # Give GPU memory time to release
    sleep 5
}

# =========================================================================
# Helper: wait for a server to be ready
# =========================================================================
wait_for_server() {
    local port=$1
    local name=$2
    echo "    Waiting for ${name} (port ${port})..."
    timeout 1200 bash -c "
        until curl -s localhost:${port}/v1/completions > /dev/null 2>&1; do
            sleep 2
        done" && echo "    ${name} is ready." || { echo "    TIMEOUT waiting for ${name}"; return 1; }
}

# =========================================================================
# Run one disaggregated benchmark
# =========================================================================
run_one() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    local TP_SIZE=$3
    local CONFIG_FILE=$4
    local BENCHMARK_NAME=$5
    local LIMIT=$6
    local TASKS=$7
    local EXTRA_ARGS=$8

    local CONFIG_NAME=$(basename "${CONFIG_FILE}" .json)
    local TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    local MODEL_BASE=$(basename "${MODEL_PATH}")
    local RESULTS_DIR="${TMP_HOME}/results/${MODEL_BASE}/${BENCHMARK_NAME}"
    local STAT_FILE="disagg_nixl_${MODEL_BASE}_${BENCHMARK_NAME}_conf_${CONFIG_NAME}_${TIMESTAMP}"

    mkdir -p "${RESULTS_DIR}"

    local PLOG="${LOGDIR}/prefill_${MODEL_NAME}_${CONFIG_NAME}_${BENCHMARK_NAME}_${TIMESTAMP}.log"
    local DLOG="${LOGDIR}/decode_${MODEL_NAME}_${CONFIG_NAME}_${BENCHMARK_NAME}_${TIMESTAMP}.log"
    local XLOG="${LOGDIR}/proxy_${MODEL_NAME}_${CONFIG_NAME}_${BENCHMARK_NAME}_${TIMESTAMP}.log"

    echo ""
    echo "    Config:    ${CONFIG_NAME}"
    echo "    Benchmark: ${BENCHMARK_NAME} (limit=${LIMIT})"
    echo "    TP:        ${TP_SIZE}"
    echo "    Timestamp: ${TIMESTAMP}"
    echo "    Results:   ${RESULTS_DIR}/${STAT_FILE}.*"

    # --- Ensure ports are free ---
    kill_ports

    export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
    local PIDS=()

    # Launch servers via setsid so each gets its own process group.
    # This lets us kill -PGID without hitting the parent script.
    # Logs go to files (no tee) to avoid pipe subshells sharing our PGID.

    # --- Launch prefill ---
    setsid bash -c "
        CUDA_VISIBLE_DEVICES=${PREFILL_GPUS} VLLM_NIXL_SIDE_CHANNEL_PORT=${PREFILL_SIDE_CHANNEL} \
        python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --host 0.0.0.0 \
        --port ${PREFILL_PORT} \
        --tensor-parallel-size ${TP_SIZE} \
        --max-model-len ${MAX_MODEL_LEN} \
        --max-num-seqs ${MAX_BATCH_SIZE} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --mixtral_config_file ${CONFIG_FILE} \
        --trust-remote-code \
        --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' \
        > '${PLOG}' 2>&1
    " &
    PIDS+=($!)

    # --- Launch decode (NO enforce-eager: CUDA graphs needed for Lynx) ---
    setsid bash -c "
        CUDA_VISIBLE_DEVICES=${DECODE_GPUS} VLLM_NIXL_SIDE_CHANNEL_PORT=${DECODE_SIDE_CHANNEL} \
        python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --host 0.0.0.0 \
        --port ${DECODE_PORT} \
        --tensor-parallel-size ${TP_SIZE} \
        --max-model-len ${MAX_MODEL_LEN} \
        --max-num-seqs ${MAX_BATCH_SIZE} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --mixtral_config_file ${CONFIG_FILE} \
        --trust-remote-code \
        --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' \
        > '${DLOG}' 2>&1
    " &
    PIDS+=($!)

    # --- Wait for both servers ---
    if ! wait_for_server ${PREFILL_PORT} "prefill"; then
        echo "    ERROR: Prefill server failed to start. Skipping."
        for pid in "${PIDS[@]}"; do kill -9 -$(ps -o sid= -p $pid 2>/dev/null | tr -d ' ') 2>/dev/null; kill -9 $pid 2>/dev/null; done || true
        kill_ports
        return 1
    fi
    if ! wait_for_server ${DECODE_PORT} "decode"; then
        echo "    ERROR: Decode server failed to start. Skipping."
        for pid in "${PIDS[@]}"; do kill -9 -$(ps -o sid= -p $pid 2>/dev/null | tr -d ' ') 2>/dev/null; kill -9 $pid 2>/dev/null; done || true
        kill_ports
        return 1
    fi

    # --- Launch proxy ---
    setsid python3 ${PROWL_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
        --port ${PROXY_PORT} \
        --prefiller-hosts localhost --prefiller-ports ${PREFILL_PORT} \
        --decoder-hosts localhost --decoder-ports ${DECODE_PORT} \
        > "${XLOG}" 2>&1 &
    PIDS+=($!)
    sleep 3

    # --- Run lm-eval ---
    export HF_ALLOW_CODE_EVAL=1
    local MODEL_ARGS="base_url=http://localhost:${PROXY_PORT}/v1/completions,add_bos_token=True,max_model_len=${MAX_MODEL_LEN},max_length=${MAX_MODEL_LEN},num_concurrent=20,timeout=1800"

    lm-eval --model local-completions \
        --tasks ${TASKS} \
        --model_args "model=${MODEL_PATH},${MODEL_ARGS}" \
        --limit ${LIMIT} \
        --log_samples --seed 42 ${EXTRA_ARGS} \
        --output_path "${RESULTS_DIR}/${STAT_FILE}.jsonl" \
        2>&1 | tee "${RESULTS_DIR}/${STAT_FILE}.log"

    local LM_EXIT=$?

    # --- Collect metrics ---
    sleep 5
    curl -s "http://localhost:${PREFILL_PORT}/metrics" > "${RESULTS_DIR}/${STAT_FILE}_prefill.metrics" 2>/dev/null || true
    curl -s "http://localhost:${DECODE_PORT}/metrics" > "${RESULTS_DIR}/${STAT_FILE}_decode.metrics" 2>/dev/null || true

    # --- Tear down all processes and free ports ---
    # Each server was launched via setsid, so it's a session leader.
    # Kill the entire session (SID) to catch TP worker children.
    echo "    Tearing down servers..."
    for pid in "${PIDS[@]}"; do
        local sid=$(ps -o sid= -p ${pid} 2>/dev/null | tr -d ' ')
        if [ -n "${sid}" ] && [ "${sid}" != "0" ]; then
            kill -9 -${sid} 2>/dev/null || true
        fi
        kill -9 "$pid" 2>/dev/null || true
    done
    kill_ports

    if [ ${LM_EXIT} -eq 0 ]; then
        echo "    DONE: ${STAT_FILE}"
    else
        echo "    WARNING: lm-eval exited with code ${LM_EXIT}"
    fi
}

# =========================================================================
# Main loop
# =========================================================================
TOTAL_MODELS=${#MODELS[@]}
TOTAL_BENCHMARKS=${#BENCHMARKS[@]}
TOTAL_RUNS=$((TOTAL_MODELS * TOTAL_BENCHMARKS * 2))  # x2 for do-nothing + prowl
RUN_NUM=0

echo "========================================="
echo "DISAGGREGATED SWEEP: ${TOTAL_MODELS} models x ${TOTAL_BENCHMARKS} benchmarks x 2 configs"
echo "  Total runs:    ${TOTAL_RUNS}"
echo "  Prefill GPUs:  ${PREFILL_GPUS}"
echo "  Decode GPUs:   ${DECODE_GPUS}"
echo "  Master log:    ${MASTER_LOG}"
echo "  Started:       $(date)"
echo "========================================="

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME MODEL_PATH TP_SIZE DO_NOTHING_CONFIG PROWL_CONFIG <<< "${model_entry}"

    echo ""
    echo "========================================="
    echo "MODEL: ${MODEL_NAME} (${MODEL_PATH}), TP=${TP_SIZE}"
    echo "========================================="

    for config in "${DO_NOTHING_CONFIG}" "${PROWL_CONFIG}"; do
        for bench_entry in "${BENCHMARKS[@]}"; do
            IFS='|' read -r BENCH_NAME LIMIT TASKS EXTRA_ARGS <<< "${bench_entry}"

            RUN_NUM=$((RUN_NUM + 1))

            if [ ${RUN_NUM} -lt ${START_FROM} ]; then
                echo "  [${RUN_NUM}/${TOTAL_RUNS}] ${MODEL_NAME} / $(basename ${config} .json) / ${BENCH_NAME} — SKIPPED (before start_from=${START_FROM})"
                continue
            fi

            echo ""
            echo "  ─────────────────────────────────────"
            echo "  [${RUN_NUM}/${TOTAL_RUNS}] ${MODEL_NAME} / $(basename ${config} .json) / ${BENCH_NAME}"
            echo "  $(date)"
            echo "  ─────────────────────────────────────"

            run_one "${MODEL_NAME}" "${MODEL_PATH}" "${TP_SIZE}" \
                    "${config}" "${BENCH_NAME}" "${LIMIT}" "${TASKS}" "${EXTRA_ARGS}" \
                || echo "  SKIPPED due to error."
        done
    done
done

echo ""
echo "========================================="
echo "ALL DISAGGREGATED SWEEP RUNS COMPLETED"
echo "  Finished: $(date)"
echo "  Total:    ${RUN_NUM}/${TOTAL_RUNS} attempted"
echo "  Log:      ${MASTER_LOG}"
echo "========================================="
