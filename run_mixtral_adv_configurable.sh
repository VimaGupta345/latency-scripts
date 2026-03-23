#!/bin/bash

export TMP_HOME="${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}"

# Activate prowl venv to use the correct vllm fork
source "${TMP_HOME}/prowl/.venv/bin/activate"

# Check if required parameters are provided
if [ $# -lt 5 ]; then
    echo "Usage: $0 <port> <model_name> <model_path> <benchmark> <limit> [config_file] [tp_size]"
    echo "Examples:"
    echo "  $0 8000 mixtral /path/to/model/ humaneval 164"
    echo "  $0 8001 mixtral /path/to/model/ gsm8k 100 /path/to/config.json 1"
    echo "  $0 8002 qwen /path/to/model/ mbpp 500 /path/to/config.json 2"
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

SERVER_ADDRESS="localhost:${PORT}"

export STATFILENAME="adv_fp16_${MODEL_NAME}_${BENCHMARK}_port${PORT}"
export MODEL="${MODEL_PATH}"

# Kill any existing vLLM server and its worker processes on this port
echo "Cleaning up port ${PORT}..."
SERVER_PIDS=$(lsof -ti:${PORT} 2>/dev/null)
if [ -n "${SERVER_PIDS}" ]; then
    for pid in ${SERVER_PIDS}; do
        # Kill the entire process group to catch TP worker children
        pgid=$(ps -o pgid= -p ${pid} 2>/dev/null | tr -d ' ')
        if [ -n "${pgid}" ]; then
            kill -9 -${pgid} 2>/dev/null
        fi
        kill -9 ${pid} 2>/dev/null
    done
fi
sleep 5

echo "Using server at: $SERVER_ADDRESS"
echo "Running benchmark: $BENCHMARK with limit: $LIMIT"
echo "Config file: $CONFIG_FILE"
echo "Tensor Parallel Size: $TP_SIZE"

# Export TP_SIZE for the serving script to use
export TP_SIZE="${TP_SIZE}"

# Run the single benchmark with specified limit and config
# Build command with optional limit flag
export TMP_HOME="${TMP_HOME}"
CMD="TMP_HOME=${TMP_HOME} python lm_eval_online_serve.py -m \"${MODEL}\" -o \"${STATFILENAME}\" -b \"${BENCHMARK}\""

# Add limit flag only if LIMIT is not empty
if [ -n "${LIMIT}" ]; then
    CMD="${CMD} -l \"${LIMIT}\""
fi

# Add remaining flags
CMD="${CMD} -k 0 -t 900 -cf \"${CONFIG_FILE}\" -sa \"${SERVER_ADDRESS}\" -mb ${MAX_BATCH_SIZE} -s \"./online_serving_ngram_port.sh\""

# Execute the command
eval ${CMD}
