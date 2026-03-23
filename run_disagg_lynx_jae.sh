#!/bin/bash
# Disaggregated prefill/decode with Lynx expert reduction.
# Uses NixlConnector for correct async KV transfer.
# Launches: prefill server -> decode server -> proxy -> benchmark or test requests.
#
# Usage:
#   bash run_disagg_lynx.sh [model] [config] [tp_size] [prefill_gpu] [decode_gpu] [benchmark]
#
# Examples:
#   # Quick smoke test (2 curl requests)
#   bash run_disagg_lynx.sh
#
#   # Qwen2-57B humaneval benchmark, TP=2
#   bash run_disagg_lynx.sh Qwen/Qwen2-57B-A14B-Instruct \
#       /data/vgupta345/prowl_related_data/prowl-open-source/prowl/configs/qwen/quant_alpha3_beta4_optimized.json \
#       2 0,1 2,3 humaneval

set -e

export TMP_HOME="${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}"
PROWL_ROOT="${TMP_HOME}/prowl"

source "${PROWL_ROOT}/.venv/bin/activate"

# --- Configurable parameters ---
MODEL=${1:-Qwen/Qwen3-30B-A3B-Instruct-2507}
CONFIG_FILE=${2:-${PROWL_ROOT}/configs/qwen3_30b/quant_alpha3_beta2_optimized.json}
TP_SIZE=${3:-1}
PREFILL_GPUS=${4:-0}
DECODE_GPUS=${5:-1}
BENCHMARK=${6:-}  # empty = smoke test, or: humaneval, gsm8k, mbpp, etc.

PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8000
PREFILL_SIDE_CHANNEL=5559
DECODE_SIDE_CHANNEL=5659
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.9
MAX_BATCH_SIZE=16

export VLLM_USE_V1=1

echo "========================================="
echo "Disaggregated Lynx Run (NixlConnector)"
echo "  Model:        ${MODEL}"
echo "  Config:       ${CONFIG_FILE}"
echo "  TP:           ${TP_SIZE}"
echo "  Prefill GPUs: ${PREFILL_GPUS} (port ${PREFILL_PORT})"
echo "  Decode GPUs:  ${DECODE_GPUS} (port ${DECODE_PORT})"
echo "  Proxy port:   ${PROXY_PORT}"
echo "  Benchmark:    ${BENCHMARK:-smoke test}"
echo "========================================="

# --- Cleanup function ---
PIDS=()
cleanup() {
    echo ""
    echo "Cleaning up..."
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    for port in ${PREFILL_PORT} ${DECODE_PORT} ${PROXY_PORT}; do
        lsof -ti:${port} 2>/dev/null | xargs kill -9 2>/dev/null || true
    done
    echo "Done."
    exit 0
}
trap cleanup INT TERM EXIT

# --- Kill anything already on our ports ---
ALL_PORTS="${PREFILL_PORT} ${DECODE_PORT} ${PROXY_PORT}"
for port in ${ALL_PORTS}; do
    lsof -ti:${port} 2>/dev/null | xargs kill -9 2>/dev/null || true
done
sleep 3

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# --- Log directory ---
LOGDIR="${TMP_HOME}/results/logs/disagg"
mkdir -p "${LOGDIR}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
PREFILL_LOG="${LOGDIR}/prefill_${TIMESTAMP}.log"
DECODE_LOG="${LOGDIR}/decode_${TIMESTAMP}.log"
PROXY_LOG="${LOGDIR}/proxy_${TIMESTAMP}.log"
echo "Logs: ${LOGDIR}/*_${TIMESTAMP}.log"

# --- Wait helper ---
wait_for_server() {
    local port=$1
    local name=$2
    echo "Waiting for ${name} (port ${port})..."
    timeout 1200 bash -c "
        until curl -s localhost:${port}/v1/completions > /dev/null 2>&1; do
            sleep 2
        done" && echo "${name} is ready." || { echo "TIMEOUT waiting for ${name}"; exit 1; }
}

# --- Launch prefill instance (kv_both) ---
echo ""
echo "Starting prefill instance on GPU ${PREFILL_GPUS}..."
CUDA_VISIBLE_DEVICES=${PREFILL_GPUS} VLLM_NIXL_SIDE_CHANNEL_PORT=${PREFILL_SIDE_CHANNEL} \
    python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --host 0.0.0.0 \
    --port ${PREFILL_PORT} \
    --tensor-parallel-size ${TP_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_BATCH_SIZE} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --mixtral_config_file ${CONFIG_FILE} \
    --trust-remote-code \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    2>&1 | tee "${PREFILL_LOG}" &
PIDS+=($!)

# --- Launch decode instance (kv_both) ---
# NOTE: Do NOT use --enforce-eager here. CUDA graphs must be enabled on the
# decode instance so that the Lynx expert-reduction routing path (captured
# during warmup with is_prefill=False, profile_complete=True) is replayed
# efficiently. Without CUDA graphs the per-step Python overhead of the
# routing function dominates and masks any memory-bandwidth savings.
echo "Starting decode instance on GPU ${DECODE_GPUS}..."
CUDA_VISIBLE_DEVICES=${DECODE_GPUS} VLLM_NIXL_SIDE_CHANNEL_PORT=${DECODE_SIDE_CHANNEL} \
    python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --host 0.0.0.0 \
    --port ${DECODE_PORT} \
    --tensor-parallel-size ${TP_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_BATCH_SIZE} \
    --gpu-memory-utilization 0.7 \
    --mixtral_config_file ${CONFIG_FILE} \
    --trust-remote-code \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    2>&1 | tee "${DECODE_LOG}" &
PIDS+=($!)

# --- Wait for both servers ---
wait_for_server ${PREFILL_PORT} "prefill"
wait_for_server ${DECODE_PORT} "decode"

# --- Launch proxy (nixl toy_proxy_server) ---
echo ""
echo "Starting proxy on port ${PROXY_PORT}..."
python3 ${PROWL_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
    --port ${PROXY_PORT} \
    --prefiller-hosts localhost --prefiller-ports ${PREFILL_PORT} \
    --decoder-hosts localhost --decoder-ports ${DECODE_PORT} \
    2>&1 | tee "${PROXY_LOG}" &
PIDS+=($!)
sleep 3

if [ -z "${BENCHMARK}" ]; then
    # =============================================
    # Smoke test: send 2 curl requests
    # =============================================
    echo ""
    echo "========================================="
    echo "Sending test requests to proxy (port ${PROXY_PORT})..."
    echo "========================================="

    output1=$(curl -X POST -s http://localhost:${PROXY_PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${MODEL}"'",
            "prompt": "San Francisco is a",
            "max_tokens": 50,
            "temperature": 0
        }')

    output2=$(curl -X POST -s http://localhost:${PROXY_PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${MODEL}"'",
            "prompt": "The theory of general relativity states that",
            "max_tokens": 50,
            "temperature": 0
        }')

    echo ""
    echo "========================================="
    echo "RESULTS"
    echo "========================================="
    echo ""
    echo "Request 1: $output1"
    echo ""
    echo "Request 2: $output2"
    echo ""
    echo "========================================="
    echo "Disaggregated Lynx smoke test complete."
    echo "========================================="
else
    # =============================================
    # Run lm-eval benchmark against the proxy
    # =============================================
    MODEL_NAME=$(basename ${MODEL})
    CONFIG_NAME=$(basename ${CONFIG_FILE} .json)
    RESULTS_DIR="${TMP_HOME}/results/${MODEL_NAME}/${BENCHMARK}"
    mkdir -p "${RESULTS_DIR}"
    STAT_FILE="disagg_nixl_${MODEL_NAME}_${BENCHMARK}_conf_${CONFIG_NAME}_${TIMESTAMP}"

    # Benchmark-specific settings
    case ${BENCHMARK} in
        humaneval)
            LIMIT=164
            TASKS="humaneval"
            EXTRA_ARGS="--trust_remote_code --confirm_run_unsafe_code"
            ;;
        mbpp)
            LIMIT=250
            TASKS="mbpp"
            EXTRA_ARGS="--trust_remote_code --confirm_run_unsafe_code --num_fewshot 3"
            ;;
        gsm8k)
            LIMIT=900
            TASKS="gsm8k"
            EXTRA_ARGS="--num_fewshot 5"
            ;;
        minerva_math_algebra)
            LIMIT=900
            TASKS="minerva_math_algebra"
            EXTRA_ARGS="--num_fewshot 4"
            ;;
        *)
            echo "Unknown benchmark: ${BENCHMARK}"
            echo "Supported: humaneval, mbpp, gsm8k, minerva_math_algebra"
            exit 1
            ;;
    esac

    echo ""
    echo "========================================="
    echo "Running ${BENCHMARK} (limit=${LIMIT}) via lm-eval"
    echo "  Results: ${RESULTS_DIR}/${STAT_FILE}.*"
    echo "========================================="

    export HF_ALLOW_CODE_EVAL=1

    MODEL_ARGS="base_url=http://localhost:${PROXY_PORT}/v1/completions,add_bos_token=True,max_model_len=${MAX_MODEL_LEN},max_length=${MAX_MODEL_LEN},num_concurrent=20,timeout=1800"

    lm-eval --model local-completions \
        --tasks ${TASKS} \
        --model_args "model=${MODEL},${MODEL_ARGS}" \
        --limit ${LIMIT} \
        --log_samples --seed 42 ${EXTRA_ARGS} \
        --output_path "${RESULTS_DIR}/${STAT_FILE}.jsonl" \
        2>&1 | tee "${RESULTS_DIR}/${STAT_FILE}.log"

    echo ""
    echo "Collecting metrics..."
    sleep 5
    curl -s "http://localhost:${PREFILL_PORT}/metrics" > "${RESULTS_DIR}/${STAT_FILE}_prefill.metrics"
    curl -s "http://localhost:${DECODE_PORT}/metrics" > "${RESULTS_DIR}/${STAT_FILE}_decode.metrics"

    echo ""
    echo "========================================="
    echo "Benchmark complete."
    echo "  lm-eval log:       ${RESULTS_DIR}/${STAT_FILE}.log"
    echo "  lm-eval results:   ${RESULTS_DIR}/${STAT_FILE}.jsonl"
    echo "  Prefill metrics:   ${RESULTS_DIR}/${STAT_FILE}_prefill.metrics"
    echo "  Decode metrics:    ${RESULTS_DIR}/${STAT_FILE}_decode.metrics"
    echo "========================================="
fi

# Cleanup is handled by the EXIT trap
