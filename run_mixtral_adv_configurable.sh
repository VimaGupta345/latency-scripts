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
ENABLE_DISAGG_SERVING=${ENABLE_DISAGG_SERVING:-false}

DISAGG_PROXY_PORT=${DISAGG_PROXY_PORT:-${PORT}}
DISAGG_PREFILL_PORT=${DISAGG_PREFILL_PORT:-$((PORT+1))}
DISAGG_DECODE_PORT=${DISAGG_DECODE_PORT:-$((PORT+2))}
DISAGG_KV_PORT=${DISAGG_KV_PORT:-14579}
DISAGG_PREFILL_GPUS=${DISAGG_PREFILL_GPUS// /}
DISAGG_DECODE_GPUS=${DISAGG_DECODE_GPUS// /}
DISAGG_KV_CONNECTOR=${DISAGG_KV_CONNECTOR:-"PyNcclConnector"}

SERVER_ADDRESS="localhost:${PORT}"
METRICS_ADDRESS=${METRICS_ADDRESS:-${SERVER_ADDRESS}}
HEALTH_ADDRESS=${HEALTH_ADDRESS:-${SERVER_ADDRESS}}
STAT_PORT=${PORT}

if [ "${ENABLE_DISAGG_SERVING}" = "true" ]; then
    SERVER_ADDRESS="localhost:${DISAGG_PROXY_PORT}"
    STAT_PORT=${DISAGG_PROXY_PORT}
    METRICS_ADDRESS=${METRICS_ADDRESS:-"localhost:${DISAGG_DECODE_PORT}"}
    HEALTH_ADDRESS=${HEALTH_ADDRESS:-${METRICS_ADDRESS}}
fi

export STATFILENAME="adv_fp16_${MODEL_NAME}_${BENCHMARK}_port${STAT_PORT}"
export MODEL="${MODEL_PATH}"
export ENABLE_DISAGG_SERVING DISAGG_PROXY_PORT DISAGG_PREFILL_PORT \
       DISAGG_DECODE_PORT DISAGG_KV_PORT DISAGG_PREFILL_GPUS \
       DISAGG_DECODE_GPUS DISAGG_KV_CONNECTOR

# Kill any existing process on this port
echo "Cleaning up ports..."
for p in ${PORT} ${DISAGG_PROXY_PORT} ${DISAGG_PREFILL_PORT} ${DISAGG_DECODE_PORT}; do
    lsof -ti:${p} | xargs -r kill -9 2>/dev/null
done
sleep 2

echo "Using server at: $SERVER_ADDRESS"
echo "Metrics address: $METRICS_ADDRESS"
echo "Health check address: $HEALTH_ADDRESS"
echo "Running benchmark: $BENCHMARK with limit: $LIMIT"
echo "Config file: $CONFIG_FILE"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Disaggregated serving: $ENABLE_DISAGG_SERVING"
if [ "${ENABLE_DISAGG_SERVING}" = "true" ]; then
    echo "  Proxy port: ${DISAGG_PROXY_PORT}, Prefill port: ${DISAGG_PREFILL_PORT}, Decode port: ${DISAGG_DECODE_PORT}, KV port: ${DISAGG_KV_PORT}"
    echo "  Prefill GPUs: ${DISAGG_PREFILL_GPUS:-<inherit CUDA_VISIBLE_DEVICES>}, Decode GPUs: ${DISAGG_DECODE_GPUS:-<inherit CUDA_VISIBLE_DEVICES>}, Connector: ${DISAGG_KV_CONNECTOR}"
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

if [ "${ENABLE_DISAGG_SERVING}" = "true" ]; then
    CMD="${CMD} --enable-disagg-serving"
    CMD="${CMD} --disagg-prefill-port ${DISAGG_PREFILL_PORT}"
    CMD="${CMD} --disagg-decode-port ${DISAGG_DECODE_PORT}"
    CMD="${CMD} --disagg-proxy-port ${DISAGG_PROXY_PORT}"
    CMD="${CMD} --disagg-kv-port ${DISAGG_KV_PORT}"
    CMD="${CMD} --disagg-kv-connector ${DISAGG_KV_CONNECTOR}"
    if [ -n "${DISAGG_PREFILL_GPUS}" ]; then
        CMD="${CMD} --disagg-prefill-gpus ${DISAGG_PREFILL_GPUS}"
    fi
    if [ -n "${DISAGG_DECODE_GPUS}" ]; then
        CMD="${CMD} --disagg-decode-gpus ${DISAGG_DECODE_GPUS}"
    fi
fi

# Execute the command
eval ${CMD}
