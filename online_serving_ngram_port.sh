#!/bin/bash

set -e

export MODEL=${1}
export STATFILENAME=${2}
export K=${3}
export MAXEXP=${4:-8}
export CONF_THRES=${5:-1.0}
export CONFIG_FILE=${6}
export PORT=${7:-8000}  # New port parameter, default to 8000
export MAX_BATCH_SIZE=${8:-16}
export ENABLE_EXPERT_PARALLEL=${9:-false}

ENABLE_DISAGG_SERVING=${ENABLE_DISAGG_SERVING:-false}
DISAGG_PROXY_PORT=${DISAGG_PROXY_PORT:-${PORT}}
DISAGG_PREFILL_PORT=${DISAGG_PREFILL_PORT:-$((PORT+1))}
DISAGG_DECODE_PORT=${DISAGG_DECODE_PORT:-$((PORT+2))}
DISAGG_KV_PORT=${DISAGG_KV_PORT:-14579}
DISAGG_KV_CONNECTOR=${DISAGG_KV_CONNECTOR:-"PyNcclConnector"}
DISAGG_KV_PARALLEL_SIZE=${DISAGG_KV_PARALLEL_SIZE:-2}
DISAGG_PREFILL_GPUS=${DISAGG_PREFILL_GPUS:-""}
DISAGG_DECODE_GPUS=${DISAGG_DECODE_GPUS:-""}
DISAGG_PREFILL_TP_SIZE=${DISAGG_PREFILL_TP_SIZE:-""}
DISAGG_DECODE_TP_SIZE=${DISAGG_DECODE_TP_SIZE:-""}
ENFORCE_EAGER=${ENFORCE_EAGER:-false}

export MODELNAME=$( basename ${MODEL} )
export CONFIGNAME=$( basename ${CONFIG_FILE} )
export STATS_DIR="/var/tmp/jae/latency-scripts/stats/perf/spec_decode/ngram/${MODELNAME}/"
mkdir -p ${STATS_DIR}
export INF_TOKS=$((K+1))
export STAT_FILE="${STATFILENAME}_${CONFIGNAME}_port${PORT}.log"

# Use TP_SIZE from environment if set, otherwise determine based on model
if [ -z "$TP_SIZE" ]; then
    export TP_SIZE=1
    if [[ $MODEL == *"Mixtral"* ]]
    then
        export TP_SIZE=1
    fi
    if [[ $MODEL == *"FP8"* ]]
    then
        export TP_SIZE=1
    fi
fi

echo "Using TP_SIZE=${TP_SIZE} for model ${MODEL}"

if [ "${ENABLE_EXPERT_PARALLEL}" = "true" ]; then
    echo "Expert parallelism ENABLED"
    EP_FLAGS="--enable-expert-parallel"
else
    echo "Expert parallelism DISABLED"
    EP_FLAGS=""
fi

if [ "${ENFORCE_EAGER}" = "true" ]; then
    echo "Enforce eager ENABLED"
    EAGER_FLAG="--enforce-eager"
    COMPILATION_ARG=""
else
    echo "Enforce eager DISABLED"
    EAGER_FLAG=""
    COMPILATION_ARG="--compilation-config '{\"full_cuda_graph\": true}'"
fi

count_gpus() {
    local gpu_list=${1// /}
    IFS=',' read -ra arr <<< "${gpu_list}"
    echo ${#arr[@]}
}

wait_for_health() {
    local port=$1
    local timeout=${2:-600}
    local start_ts=$(date +%s)
    while true; do
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        if [ $(( $(date +%s) - start_ts )) -ge ${timeout} ]; then
            echo "Timed out waiting for server on port ${port}"
            return 1
        fi
        sleep 2
    done
}

cleanup() {
    for pid in ${PREFILL_PID:-} ${DECODE_PID:-} ${PROXY_PID:-}; do
        if [ -n "${pid}" ]; then
            kill "${pid}" >/dev/null 2>&1 || true
        fi
    done
}

if [ "${ENABLE_DISAGG_SERVING}" = "true" ]; then
    trap cleanup EXIT

    CUDA_CLEAN=${CUDA_VISIBLE_DEVICES// /}
    if [ -z "${DISAGG_PREFILL_GPUS}" ] || [ -z "${DISAGG_DECODE_GPUS}" ]; then
        IFS=',' read -ra CUDA_GPUS <<< "${CUDA_CLEAN}"
        if [ -z "${DISAGG_PREFILL_GPUS}" ] && [ ${#CUDA_GPUS[@]} -ge 1 ]; then
            DISAGG_PREFILL_GPUS=${CUDA_GPUS[0]}
        fi
        if [ -z "${DISAGG_DECODE_GPUS}" ] && [ ${#CUDA_GPUS[@]} -ge 2 ]; then
            DISAGG_DECODE_GPUS=${CUDA_GPUS[1]}
        fi
    fi

    if [ -z "${DISAGG_PREFILL_GPUS}" ] || [ -z "${DISAGG_DECODE_GPUS}" ]; then
        echo "Disaggregated serving requested but DISAGG_PREFILL_GPUS and DISAGG_DECODE_GPUS are not set (and CUDA_VISIBLE_DEVICES does not provide two GPUs)."
        exit 1
    fi

    PREFILL_TP=${DISAGG_PREFILL_TP_SIZE}
    if [ -z "${PREFILL_TP}" ]; then
        PREFILL_TP=$(count_gpus "${DISAGG_PREFILL_GPUS}")
    fi
    DECODE_TP=${DISAGG_DECODE_TP_SIZE}
    if [ -z "${DECODE_TP}" ]; then
        DECODE_TP=$(count_gpus "${DISAGG_DECODE_GPUS}")
    fi

    echo "Starting disaggregated serving:"
    echo "  Proxy port: ${DISAGG_PROXY_PORT}, Prefill port: ${DISAGG_PREFILL_PORT}, Decode port: ${DISAGG_DECODE_PORT}"
    echo "  Prefill GPUs: ${DISAGG_PREFILL_GPUS} (TP=${PREFILL_TP}), Decode GPUs: ${DISAGG_DECODE_GPUS} (TP=${DECODE_TP})"
    echo "  KV connector: ${DISAGG_KV_CONNECTOR}, KV port: ${DISAGG_KV_PORT}, KV parallel size: ${DISAGG_KV_PARALLEL_SIZE}"

    for p in ${DISAGG_PROXY_PORT} ${DISAGG_PREFILL_PORT} ${DISAGG_DECODE_PORT}; do
        lsof -ti:${p} | xargs -r kill -9 2>/dev/null || true
    done

    PREFILL_LOG="${STATS_DIR}/${STATFILENAME}_${CONFIGNAME}_prefill_port${DISAGG_PREFILL_PORT}.log"
    DECODE_LOG="${STATS_DIR}/${STATFILENAME}_${CONFIGNAME}_decode_port${DISAGG_DECODE_PORT}.log"
    PROXY_LOG="${STATS_DIR}/${STATFILENAME}_${CONFIGNAME}_proxy_port${DISAGG_PROXY_PORT}.log"

    PREFILL_CMD="CUDA_VISIBLE_DEVICES=${DISAGG_PREFILL_GPUS} python -m vllm.entrypoints.openai.api_server --model ${MODEL} \
    --host localhost \
    --port ${DISAGG_PREFILL_PORT} \
    --tensor-parallel-size ${PREFILL_TP} \
    --max-num-seqs ${MAX_BATCH_SIZE} \
    --max-model-len 4096 \
    ${COMPILATION_ARG} \
    --gpu-memory-utilization 0.9 \
    --mixtral_config_file ${CONFIG_FILE} \
    --trust-remote-code \
    ${EAGER_FLAG} \
    --kv-transfer-config '{\"kv_connector\":\"${DISAGG_KV_CONNECTOR}\",\"kv_role\":\"kv_producer\",\"kv_rank\":0,\"kv_parallel_size\":${DISAGG_KV_PARALLEL_SIZE},\"kv_port\":${DISAGG_KV_PORT}}' \
    ${EP_FLAGS}"

    if [ $K -eq 0 ]; then
        DECODE_BATCH_FLAGS="--max-num-seqs ${MAX_BATCH_SIZE} ${COMPILATION_ARG} --gpu-memory-utilization 0.9 --max-model-len 4096 ${EAGER_FLAG}"
        DECODE_SPEC_FLAGS=""
        export MIDAS_ENABLE=false
    else
        if [ "$MAXEXP" != "8" ]; then
            export MIDAS_ENABLE=true
            export MIDAS_INFLIGHT_TOKS=${INF_TOKS}
            export MIDAS_MAX_EXPERTS=${MAXEXP}
            export MIDAS_CONF_THRES=${CONF_THRES}
        else
            export MIDAS_ENABLE=false
        fi
        if [[ $STATFILENAME == *"casd"* ]]; then
            export MIDAS_CASD_ENABLE=true
        else
            export MIDAS_CASD_ENABLE=false
        fi
        echo MIDAS_ENABLE=${MIDAS_ENABLE}, MIDAS_INFLIGHT_TOKS=${MIDAS_INFLIGHT_TOKS}
        echo MIDAS_MAX_EXPERTS=${MIDAS_MAX_EXPERTS}, MIDAS_CONF_THRES=${MIDAS_CONF_THRES}
        DECODE_BATCH_FLAGS="--max-num-seqs 1 --max-num-batched-tokens 4096 --max-model-len 4096 --gpu-memory-utilization 0.99 --enforce-eager"
        DECODE_SPEC_FLAGS="--speculative-model [ngram] --speculative-draft-tensor-parallel-size 1 --num-speculative-tokens ${K} --ngram-prompt-lookup-max $((K*2))"
    fi

    DECODE_CMD="CUDA_VISIBLE_DEVICES=${DISAGG_DECODE_GPUS} python -m vllm.entrypoints.openai.api_server --model ${MODEL} \
    --host localhost \
    --port ${DISAGG_DECODE_PORT} \
    --tensor-parallel-size ${DECODE_TP} \
    ${DECODE_BATCH_FLAGS} \
    --mixtral_config_file ${CONFIG_FILE} \
    --trust-remote-code \
    --kv-transfer-config '{\"kv_connector\":\"${DISAGG_KV_CONNECTOR}\",\"kv_role\":\"kv_consumer\",\"kv_rank\":1,\"kv_parallel_size\":${DISAGG_KV_PARALLEL_SIZE},\"kv_port\":${DISAGG_KV_PORT}}' \
    ${DECODE_SPEC_FLAGS} \
    ${EP_FLAGS}"

    bash -c "${PREFILL_CMD}" > "${PREFILL_LOG}" 2>&1 &
    PREFILL_PID=$!
    bash -c "${DECODE_CMD}" > "${DECODE_LOG}" 2>&1 &
    DECODE_PID=$!

    wait_for_health ${DISAGG_PREFILL_PORT} || true
    wait_for_health ${DISAGG_DECODE_PORT} || true

    PROXY_SCRIPT="${TMP_HOME:-/var/tmp/jae}/prowl/benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py"
    PROXY_DIR=$(dirname "${PROXY_SCRIPT}")
    if [ ! -f "${PROXY_SCRIPT}" ]; then
        echo "Proxy script not found at ${PROXY_SCRIPT}"
        exit 1
    fi
    (cd "${PROXY_DIR}" && PYTHONPATH="${PROXY_DIR}:${PYTHONPATH:-}" python disagg_prefill_proxy_server.py \
        --port ${DISAGG_PROXY_PORT} \
        --prefill-url "http://localhost:${DISAGG_PREFILL_PORT}/v1/completions" \
        --decode-url "http://localhost:${DISAGG_DECODE_PORT}/v1/completions") > "${PROXY_LOG}" 2>&1 &
    PROXY_PID=$!

    wait ${PREFILL_PID} ${DECODE_PID} ${PROXY_PID}
    exit 0
fi

echo "Starting vLLM server on port ${PORT}"
#--compilation-config '{"full_cuda_graph": true}' \
# if value of K is 0, then don't use speculative model 
if [ $K -eq 0 ]
then
    echo "baseline run"
    echo "midas disabled"
    export MIDAS_ENABLE=false
    python -m vllm.entrypoints.openai.api_server --model ${MODEL} \
    --host localhost \
    --port ${PORT} \
    --max-num-seqs ${MAX_BATCH_SIZE} \
    --tensor-parallel-size ${TP_SIZE} \
    --max-model-len 4096 \
    ${COMPILATION_ARG} \
    --gpu-memory-utilization 0.9  \
    --mixtral_config_file ${CONFIG_FILE} \
    --trust-remote-code \
    ${EAGER_FLAG} \
    ${EP_FLAGS} \
    2>&1 | tee ${STATS_DIR}/${STAT_FILE}
    exit 0
else
    echo "speculative ngram run"
    if [ "$MAXEXP" != "8" ]
    then
        echo "midas enabled"
        export MIDAS_ENABLE=true
        export MIDAS_INFLIGHT_TOKS=${INF_TOKS}
        export MIDAS_MAX_EXPERTS=${MAXEXP}
        export MIDAS_CONF_THRES=${CONF_THRES}
    else
        echo "midas disabled"
        export MIDAS_ENABLE=false
    fi
    # if casd is in STATFILENAME, enable casd
    if [[ $STATFILENAME == *"casd"* ]]
    then
        echo "midas casd enabled"
        export MIDAS_CASD_ENABLE=true
    else
        export MIDAS_CASD_ENABLE=false
    fi
    echo MIDAS_ENABLE=${MIDAS_ENABLE}, MIDAS_INFLIGHT_TOKS=${MIDAS_INFLIGHT_TOKS}
    echo MIDAS_MAX_EXPERTS=${MIDAS_MAX_EXPERTS}, MIDAS_CONF_THRES=${MIDAS_CONF_THRES}
    python -m vllm.entrypoints.openai.api_server --model ${MODEL} \
    --host localhost \
    --port ${PORT} \
    --tensor-parallel-size ${TP_SIZE} \
    --max-num-seqs 1 --max-num-batched-tokens 4096 --max-model-len 4096 \
    --gpu-memory-utilization 0.99 --enforce-eager \
    --speculative-model [ngram] --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens ${K} --ngram-prompt-lookup-max $((K*2)) \
    --trust-remote-code \
    ${EP_FLAGS} \
    2>&1 | tee ${STATS_DIR}/${STAT_FILE}
fi
