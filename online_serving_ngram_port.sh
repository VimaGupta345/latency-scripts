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
export ENABLE_EPLB=${10:-false}
export ENABLE_VLLM_EPLB=${11:-${ENABLE_VLLM_EPLB:-false}}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

ENFORCE_EAGER=${ENFORCE_EAGER:-false}
# Allow override of cudagraph mode via env.
CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-""}
REASONING_PARSER=${REASONING_PARSER:-""}
MM_ENCODER_TP_MODE=${MM_ENCODER_TP_MODE:-""}

# Sensible defaults for Qwen3-Omni if caller did not set them.
if [[ -z "${REASONING_PARSER}" && "${MODEL}" == *"Qwen3-Omni"* ]]; then
    REASONING_PARSER="qwen3"
fi
if [[ -z "${MM_ENCODER_TP_MODE}" && "${MODEL}" == *"Qwen3-Omni"* ]]; then
    MM_ENCODER_TP_MODE="data"
fi

export MODELNAME=$( basename ${MODEL} )
export CONFIGNAME=$( basename ${CONFIG_FILE} )
PERF_STATS_ROOT=${PERF_STATS_ROOT:-"/nethome/rdudala3/latency-scripts/stats/perf"}
export STATS_DIR="${PERF_STATS_ROOT}/spec_decode/ngram/${MODELNAME}/"
mkdir -p "${STATS_DIR}"
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

if [ "${ENABLE_EPLB}" = "true" ]; then
    if [ "${ENABLE_EXPERT_PARALLEL}" != "true" ]; then
        echo "Lynx EPLB requires expert parallelism. Set ENABLE_EXPERT_PARALLEL=true." >&2
        exit 1
    fi
    echo "Lynx EP load correction (lynx_routing EPLB) ENABLED"
else
    echo "Lynx EP load correction (lynx_routing EPLB) DISABLED"
fi
export LYNX_ENABLE_EPLB="${ENABLE_EPLB}"

if [ "${ENABLE_VLLM_EPLB}" = "true" ]; then
    if [ "${ENABLE_EXPERT_PARALLEL}" != "true" ]; then
        echo "vLLM native EPLB requires expert parallelism. Set ENABLE_EXPERT_PARALLEL=true." >&2
        exit 1
    fi
    echo "vLLM native EPLB ENABLED"
    VLLM_EPLB_FLAGS="--enable-eplb"
else
    echo "vLLM native EPLB DISABLED"
    VLLM_EPLB_FLAGS=""
fi

if [ "${ENFORCE_EAGER}" = "true" ]; then
    echo "Enforce eager ENABLED"
    EAGER_FLAG="--enforce-eager"
    COMPILATION_ARGS=()
else
    echo "Enforce eager DISABLED"
    EAGER_FLAG=""
    # Default to PIECEWISE for Llama-4 (FULL is not supported with CLA backend).
    if [ -z "${CUDAGRAPH_MODE}" ]; then
        if [[ "$MODEL" == *"Llama-4"* ]]; then
            CUDAGRAPH_MODE="PIECEWISE"
        else
            CUDAGRAPH_MODE="FULL"
        fi
    fi
    COMPILATION_JSON="{\"cudagraph_mode\": \"${CUDAGRAPH_MODE}\"}"
    COMPILATION_ARGS=(--compilation-config "$COMPILATION_JSON")
fi

OPTIONAL_MODEL_FLAGS=()
if [ -n "${REASONING_PARSER}" ]; then
    OPTIONAL_MODEL_FLAGS+=(--reasoning-parser "${REASONING_PARSER}")
fi
if [ -n "${MM_ENCODER_TP_MODE}" ]; then
    OPTIONAL_MODEL_FLAGS+=(--mm-encoder-tp-mode "${MM_ENCODER_TP_MODE}")
fi
ATTENTION_BACKEND_OVERRIDE=${ATTENTION_BACKEND:-${VLLM_ATTENTION_BACKEND:-""}}
if [ -n "${ATTENTION_BACKEND_OVERRIDE}" ]; then
    OPTIONAL_MODEL_FLAGS+=(--attention-backend "${ATTENTION_BACKEND_OVERRIDE}")
    echo "Attention backend override: ${ATTENTION_BACKEND_OVERRIDE}"
fi
if [ -n "${VLLM_PROFILER_CONFIG_JSON:-}" ]; then
    OPTIONAL_MODEL_FLAGS+=(--profiler-config "${VLLM_PROFILER_CONFIG_JSON}")
    echo "Profiler config override enabled"
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
    --max-model-len ${MAX_MODEL_LEN} \
    "${COMPILATION_ARGS[@]}" \
    --gpu-memory-utilization 0.9  \
    --mixtral-config-file ${CONFIG_FILE} \
    "${OPTIONAL_MODEL_FLAGS[@]}" \
    --trust-remote-code \
    ${EAGER_FLAG} \
    ${EP_FLAGS} \
    ${VLLM_EPLB_FLAGS} \
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
    "${OPTIONAL_MODEL_FLAGS[@]}" \
    --trust-remote-code \
    ${EP_FLAGS} \
    ${VLLM_EPLB_FLAGS} \
    2>&1 | tee ${STATS_DIR}/${STAT_FILE}
fi
