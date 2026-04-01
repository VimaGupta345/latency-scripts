#!/bin/bash

# Run benchmarks for thinking/reasoning models (e.g. Qwen3-235B-A22B-Thinking)
#
# Differences from run_*_4bmk.sh:
#   - Uses --reasoning-parser qwen3 on the server
#   - Uses local-chat-completions + --apply_chat_template on the eval side
#   - Uses /v1/chat/completions endpoint instead of /v1/completions
#   - Larger max_model_len (40960) and max_gen_toks (32768) for thinking chains
#   - Temperature=0.6, top_p=0.95 as recommended by Qwen3
#   - Uses math benchmarks (aime24, gsm8k, minerva_math_algebra) where
#     stop sequences don't interfere with thinking content
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_thinking_model_4bmk.sh

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
source "${TMP_HOME}/prowl/.venv/bin/activate"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
MODEL_NAME="${MODEL_NAME:-qwen3_235b_thinking}"
PORT="${PORT:-8019}"
TP_SIZE="${TP_SIZE:-4}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export VLLM_USE_V1=1

MAX_MODEL_LEN=131072
MAX_BATCH_SIZE=16

LOGDIR="${TMP_HOME}/results/logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/${MODEL_NAME}_4bmk_$(date +%Y%m%d-%H%M%S).log"

exec > >(tee -a "${LOGFILE}") 2>&1

configs=(
    "${TMP_HOME}/prowl/configs/qwen3_235b/qwen_do-nothing.json"
    "${TMP_HOME}/prowl/configs/qwen3_235b/quant_alpha3_beta2_optimized.json"
)

# Math benchmarks — stop sequences won't fire during thinking
# Format: "name|limit|tasks|extra_args"
benchmarks=(
    "aime24||aime24|"
    "gsm8k||gsm8k|--num_fewshot 5"
    "minerva_math_algebra||minerva_math_algebra|--num_fewshot 4"
)

echo "Starting thinking model benchmarks"
echo "  Model: ${MODEL_PATH}"
echo "  GPUs: ${GPUS}, TP: ${TP_SIZE}, Port: ${PORT}"
echo "  Max model len: ${MAX_MODEL_LEN}"
echo "  Log: ${LOGFILE}"
echo "========================================="

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .json)
    echo ""
    echo "Config: $config_name"
    echo "-----------------------------------------"

    for bench_entry in "${benchmarks[@]}"; do
        IFS='|' read -r BENCH_NAME LIMIT TASKS EXTRA_ARGS <<< "${bench_entry}"

        echo "  Running $BENCH_NAME with $config_name"

        MODEL_BASE=$(basename "${MODEL_PATH}")
        RESULTS_DIR="${TMP_HOME}/results/${MODEL_BASE}/${BENCH_NAME}"
        mkdir -p "${RESULTS_DIR}"

        TIMESTAMP=$(date +%Y%m%d-%H%M%S)
        STAT_FILE="thinking_${MODEL_NAME}_${BENCH_NAME}_port${PORT}_n${LIMIT}_conf_${config_name}"

        # --- Kill any existing server on this port ---
        SERVER_PIDS=$(lsof -ti:${PORT} 2>/dev/null)
        if [ -n "${SERVER_PIDS}" ]; then
            for pid in ${SERVER_PIDS}; do
                pgid=$(ps -o pgid= -p ${pid} 2>/dev/null | tr -d ' ')
                if [ -n "${pgid}" ]; then
                    kill -9 -${pgid} 2>/dev/null
                fi
                kill -9 ${pid} 2>/dev/null
            done
        fi
        sleep 5

        # --- Collect pre-run metrics ---
        export STATFILENAME="${STAT_FILE}"
        export MODEL="${MODEL_PATH}"

        # --- Start vLLM server with reasoning parser ---
        echo "Starting vLLM server with reasoning parser"
        serving_cmd="bash -c './online_serving_thinking_port.sh ${MODEL_PATH} ${STAT_FILE} 0 8 1.0 ${config} ${PORT} ${MAX_BATCH_SIZE} ${MAX_MODEL_LEN} > /dev/null 2>&1'"
        echo "serving_cmd: ${serving_cmd}"
        eval "setsid ${serving_cmd} &"
        SERVER_PID=$!

        # --- Wait for server ---
        echo "Waiting for server at http://localhost:${PORT}/health (timeout=900s)..."
        SECONDS=0
        until curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
            if [ $SECONDS -ge 900 ]; then
                echo "    TIMEOUT waiting for server"
                kill -9 -$(ps -o pgid= -p ${SERVER_PID} 2>/dev/null | tr -d ' ') 2>/dev/null
                continue 2
            fi
            sleep 10
        done
        echo "Server is ready after ${SECONDS}s"

        # --- Run lm-eval with chat completions + thinking ---
        export HF_ALLOW_CODE_EVAL=1
        MODEL_ARGS="model=${MODEL_PATH},base_url=http://localhost:${PORT}/v1/chat/completions,num_concurrent=20,max_gen_toks=65536,tokenizer_backend=huggingface,timeout=3600"

        eval_cmd="lm-eval --model local-chat-completions \
            --tasks ${TASKS} \
            --model_args ${MODEL_ARGS} \
            --apply_chat_template \
            ${LIMIT:+--limit ${LIMIT}} --log_samples --seed 42 ${EXTRA_ARGS} \
            --gen_kwargs '{\"max_gen_toks\": 65536, \"temperature\": 0.6, \"top_p\": 0.95, \"do_sample\": true}' \
            --output_path ${RESULTS_DIR}/${STAT_FILE}.jsonl \
            2>&1 | tee ${RESULTS_DIR}/${STAT_FILE}.log"

        echo "eval_cmd: ${eval_cmd}"
        eval "${eval_cmd}"

        # --- Collect post-eval metrics (sleep to let server flush) ---
        echo "collecting metrics for ${STAT_FILE}"
        sleep 5
        curl -s "http://localhost:${PORT}/metrics" > "${RESULTS_DIR}/${STAT_FILE}_${TIMESTAMP}.metrics" 2>/dev/null

        # --- Kill server ---
        echo "Killing server..."
        pgid=$(ps -o pgid= -p ${SERVER_PID} 2>/dev/null | tr -d ' ')
        if [ -n "${pgid}" ]; then
            kill -INT -${pgid} 2>/dev/null
            sleep 3
            kill -9 -${pgid} 2>/dev/null
        fi
        kill -9 ${SERVER_PID} 2>/dev/null
        sleep 5

        echo "  Completed: $BENCH_NAME with $config_name"
    done
    echo "-----------------------------------------"
done

echo ""
echo "========================================="
echo "Thinking model experiment completed!"
echo "========================================="
