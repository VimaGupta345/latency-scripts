#!/bin/bash

# Quick AIME24 run: baseline + alpha3_beta2 for Qwen3-235B thinking mode

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
source "${TMP_HOME}/prowl/.venv/bin/activate"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
MODEL_NAME="${MODEL_NAME:-qwen3_235b_thinking}"
PORT="${PORT:-8019}"
TP_SIZE="${TP_SIZE:-4}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export VLLM_USE_V1=1
export HF_ALLOW_CODE_EVAL=1

MAX_MODEL_LEN=131072
MAX_BATCH_SIZE=16

LOGDIR="${TMP_HOME}/results/logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/${MODEL_NAME}_aime24_$(date +%Y%m%d-%H%M%S).log"

exec > >(tee -a "${LOGFILE}") 2>&1

configs=(
    "${TMP_HOME}/prowl/configs/qwen3_235b/quant_alpha5_beta8_optimized.json"
)

echo "AIME24: Qwen3-235B thinking mode"
echo "  GPUs: ${GPUS}, TP: ${TP_SIZE}, Port: ${PORT}"
echo "  max_model_len: ${MAX_MODEL_LEN}, max_num_seqs: ${MAX_BATCH_SIZE}"
echo "  Log: ${LOGFILE}"
echo "========================================="

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .json)
    echo ""
    echo "Config: $config_name"
    echo "-----------------------------------------"

    MODEL_BASE=$(basename "${MODEL_PATH}")
    RESULTS_DIR="${TMP_HOME}/results/${MODEL_BASE}/aime24"
    mkdir -p "${RESULTS_DIR}"

    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    STAT_FILE="thinking_${MODEL_NAME}_aime24_port${PORT}_conf_${config_name}"

    # Kill any existing server
    for pid in $(lsof -ti:${PORT} 2>/dev/null); do
        pgid=$(ps -o pgid= -p ${pid} 2>/dev/null | tr -d ' ')
        [ -n "${pgid}" ] && kill -9 -${pgid} 2>/dev/null
        kill -9 ${pid} 2>/dev/null
    done
    sleep 5

    # Start server
    echo "Starting vLLM server with reasoning parser"
    setsid bash -c "./online_serving_thinking_port.sh ${MODEL_PATH} ${STAT_FILE} 0 8 1.0 ${config} ${PORT} ${MAX_BATCH_SIZE} ${MAX_MODEL_LEN} > /dev/null 2>&1" &
    SERVER_PID=$!

    # Wait for server
    echo "Waiting for server (timeout=900s)..."
    SECONDS=0
    until curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        if [ $SECONDS -ge 900 ]; then
            echo "TIMEOUT"
            kill -9 -$(ps -o pgid= -p ${SERVER_PID} 2>/dev/null | tr -d ' ') 2>/dev/null
            continue 2
        fi
        sleep 10
    done
    echo "Server ready after ${SECONDS}s"

    # Run eval
    MODEL_ARGS="model=${MODEL_PATH},base_url=http://localhost:${PORT}/v1/chat/completions,num_concurrent=20,max_gen_toks=65536,tokenizer_backend=huggingface,timeout=3600"

    lm-eval --model local-chat-completions \
        --tasks aime24 \
        --model_args ${MODEL_ARGS} \
        --apply_chat_template \
        --log_samples --seed 42 \
        --gen_kwargs '{"max_gen_toks": 65536, "temperature": 0.6, "top_p": 0.95, "do_sample": true}' \
        --output_path ${RESULTS_DIR}/${STAT_FILE}.jsonl \
        2>&1 | tee ${RESULTS_DIR}/${STAT_FILE}.log

    # Collect metrics
    echo "collecting metrics for ${STAT_FILE}"
    sleep 5
    curl -s "http://localhost:${PORT}/metrics" > "${RESULTS_DIR}/${STAT_FILE}_${TIMESTAMP}.metrics" 2>/dev/null

    # Kill server
    echo "Killing server..."
    pgid=$(ps -o pgid= -p ${SERVER_PID} 2>/dev/null | tr -d ' ')
    [ -n "${pgid}" ] && kill -INT -${pgid} 2>/dev/null
    sleep 3
    [ -n "${pgid}" ] && kill -9 -${pgid} 2>/dev/null
    kill -9 ${SERVER_PID} 2>/dev/null
    sleep 5

    echo "Completed: $config_name"
done

echo ""
echo "========================================="
echo "AIME24 done!"
echo "========================================="
