#!/bin/bash

# Check if required parameters are provided
if [ $# -lt 5 ]; then
    echo "Usage: $0 <port> <model_name> <model_path> <benchmark> <limit> [config_file] [tp_size] [enable_expert_parallel]"
    echo "Examples:"
    echo "  $0 8000 mixtral /path/to/model/ humaneval 164"
    echo "  $0 8001 mixtral /path/to/model/ gsm8k 100 /path/to/config.json 1"
    echo "  $0 8002 qwen /path/to/model/ mbpp 500 /path/to/config.json 2 16 true"
    echo ""
    echo "Available benchmarks: humaneval, gsm8k, mbpp, minerva_math_algebra"
    exit 1
fi

PORT=$1
MODEL_NAME=$2
MODEL_PATH=$3
BENCHMARK=$4
LIMIT=$5
CONFIG_FILE=${6:-"${TMP_HOME}/prowl/configs/mixtral/do_nothing.json"}  # Optional config, default to do_nothing
TP_SIZE=${7:-1}  # Optional TP size, default to 1
MAX_BATCH_SIZE=${8:-16}
ENABLE_EXPERT_PARALLEL=${9:-false}

SERVER_ADDRESS="localhost:${PORT}"
METRICS_ADDRESS=${METRICS_ADDRESS:-${SERVER_ADDRESS}}
HEALTH_ADDRESS=${HEALTH_ADDRESS:-${SERVER_ADDRESS}}
STAT_PORT=${PORT}

export STATFILENAME="adv_fp16_${MODEL_NAME}_${BENCHMARK}_port${STAT_PORT}"
export MODEL="${MODEL_PATH}"

# lm-eval API client timeout is 300s by default in the harness, which is too low
# for long-thinking / long-generation runs (e.g. Omni AIME). Allow override.
export LM_EVAL_TIMEOUT=${LM_EVAL_TIMEOUT:-1800}

# Kill any existing process on this port
echo "Cleaning up ports..."
for p in ${PORT}; do
    lsof -ti:${p} | xargs -r kill -9 2>/dev/null
done
sleep 2

echo "Using server at: $SERVER_ADDRESS"
echo "Metrics address: $METRICS_ADDRESS"
echo "Health check address: $HEALTH_ADDRESS"
echo "Running benchmark: $BENCHMARK with limit: $LIMIT"
echo "Config file: $CONFIG_FILE"
echo "Tensor Parallel Size: $TP_SIZE"
echo "lm-eval timeout (s): ${LM_EVAL_TIMEOUT}"
if [ -n "${LM_EVAL_GEN_KWARGS:-}" ]; then
    echo "lm-eval gen kwargs: ${LM_EVAL_GEN_KWARGS}"
fi
if [ -n "${LM_EVAL_FORCE_CHAT_COMPLETIONS:-}" ]; then
    echo "Force chat completions: ${LM_EVAL_FORCE_CHAT_COMPLETIONS}"
fi
if [ -n "${REASONING_PARSER:-}" ]; then
    echo "Reasoning parser: ${REASONING_PARSER}"
fi
if [ -n "${MM_ENCODER_TP_MODE:-}" ]; then
    echo "MM encoder TP mode: ${MM_ENCODER_TP_MODE}"
fi

# Export sizes for the serving script to use
export TP_SIZE="${TP_SIZE}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL}"

# Run the single benchmark with specified limit and config
# Build command with optional limit flag
CMD="TMP_HOME=${TMP_HOME} python lm_eval_online_serve.py -m \"${MODEL}\" -o \"${STATFILENAME}\" -b \"${BENCHMARK}\""

# Add limit flag only if LIMIT is not empty
if [ -n "${LIMIT}" ]; then
    CMD="${CMD} -l \"${LIMIT}\""
fi

# Add remaining flags
CMD="${CMD} -k 0 -t 150 -cf \"${CONFIG_FILE}\" -sa \"${SERVER_ADDRESS}\" -mb ${MAX_BATCH_SIZE} -s \"./online_serving_ngram_port.sh\" --metrics-address \"${METRICS_ADDRESS}\" --health-address \"${HEALTH_ADDRESS}\""

if [ "${ENABLE_EXPERT_PARALLEL}" = "true" ]; then
    CMD="${CMD} --enable-expert-parallel"
fi

# Execute the command
eval ${CMD}
