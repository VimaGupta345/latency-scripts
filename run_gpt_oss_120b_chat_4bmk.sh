#!/bin/bash

# GPT-OSS-120B: baseline + prowl on 4 benchmarks
#
# Uses local-chat-completions + --apply_chat_template because gpt-oss
# requires the harmony chat template for correct behavior.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 ./run_gpt_oss_120b_chat_4bmk.sh

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
source "${TMP_HOME}/prowl/.venv/bin/activate"

MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-120b}"
MODEL_NAME="${MODEL_NAME:-gpt_oss_120b}"
PORT="${PORT:-8050}"
TP_SIZE="${TP_SIZE:-4}"
GPUS="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

MAX_MODEL_LEN=4096
MAX_BATCH_SIZE=16

LOGDIR="${TMP_HOME}/results/logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/${MODEL_NAME}_chat_4bmk_$(date +%Y%m%d-%H%M%S).log"

exec > >(tee -a "${LOGFILE}") 2>&1

configs=(
    "${TMP_HOME}/prowl/configs/gpt_oss_120b/gpt_oss_do-nothing.json"
    "${TMP_HOME}/prowl/configs/gpt_oss_120b/quant_alpha3_beta2_optimized.json"
)

# Format: "name|limit|tasks|extra_args"
benchmarks=(
    "humaneval|164|humaneval|--trust_remote_code --confirm_run_unsafe_code"
    "mbpp|250|mbpp|--trust_remote_code --confirm_run_unsafe_code --num_fewshot 3"
    "gsm8k|900|gsm8k|--num_fewshot 5"
    "minerva_math_algebra|900|minerva_math_algebra|--num_fewshot 4"
)

echo "Starting GPT-OSS-120B chat-template benchmarks"
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
        STAT_FILE="chat_${MODEL_NAME}_${BENCH_NAME}_port${PORT}_n${LIMIT}_conf_${config_name}"

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

        # --- Start vLLM server ---
        MODELNAME=$(basename ${MODEL_PATH})
        STATS_DIR_SRV="${TMP_HOME}/results/server_logs/${MODELNAME}"
        mkdir -p "${STATS_DIR_SRV}"
        SRV_LOG="${STATS_DIR_SRV}/${STAT_FILE}_${TIMESTAMP}.log"

        echo "Starting vLLM server"
        setsid python -m vllm.entrypoints.openai.api_server \
            --model ${MODEL_PATH} \
            --host localhost \
            --port ${PORT} \
            --max-num-seqs ${MAX_BATCH_SIZE} \
            --tensor-parallel-size ${TP_SIZE} \
            --max-model-len ${MAX_MODEL_LEN} \
            --gpu-memory-utilization 0.9 \
            --mixtral_config_file ${config} \
            --trust-remote-code \
            2>&1 | tee "${SRV_LOG}" &
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

        # --- Run lm-eval with chat completions ---
        export HF_ALLOW_CODE_EVAL=1
        MODEL_ARGS="model=${MODEL_PATH},base_url=http://localhost:${PORT}/v1/chat/completions,num_concurrent=20,max_gen_toks=4096,tokenizer_backend=huggingface,timeout=1800"

        eval_cmd="lm-eval --model local-chat-completions \
            --tasks ${TASKS} \
            --model_args ${MODEL_ARGS} \
            --apply_chat_template \
            ${LIMIT:+--limit ${LIMIT}} --log_samples --seed 42 ${EXTRA_ARGS} \
            --output_path ${RESULTS_DIR}/${STAT_FILE}.jsonl \
            2>&1 | tee ${RESULTS_DIR}/${STAT_FILE}.log"

        echo "eval_cmd: ${eval_cmd}"
        eval "${eval_cmd}"

        # --- Collect metrics ---
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
echo "GPT-OSS-120B chat experiment completed!"
echo "  python gather_results.py results/gpt-oss-120b/"
echo "========================================="
