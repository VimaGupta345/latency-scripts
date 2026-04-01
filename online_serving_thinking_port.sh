#!/bin/bash

# Serving script for thinking/reasoning models.
# Same as online_serving_ngram_port.sh but adds --reasoning-parser qwen3
# and uses larger max-model-len for long thinking chains.

source "${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}/prowl/.venv/bin/activate"

export MODEL=${1}
export STATFILENAME=${2}
export K=${3}
export MAXEXP=${4:-8}
export CONF_THRES=${5:-1.0}
export CONFIG_FILE=${6}
export PORT=${7:-8000}
export MAX_BATCH_SIZE=${8:-4}
export MAX_MODEL_LEN=${9:-40960}

export MODELNAME=$( basename ${MODEL} )
export CONFIGNAME=$( basename ${CONFIG_FILE} )
export STATS_DIR="${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}/results/server_logs/${MODELNAME}"
mkdir -p ${STATS_DIR}
export STAT_FILE="${STATFILENAME}_${CONFIGNAME}_port${PORT}.log"

# Use TP_SIZE from environment if set
if [ -z "$TP_SIZE" ]; then
    export TP_SIZE=4
fi

echo "Using TP_SIZE=${TP_SIZE} for model ${MODEL}"
echo "Starting vLLM server with reasoning parser on port ${PORT}"

python -m vllm.entrypoints.openai.api_server --model ${MODEL} \
    --host localhost \
    --port ${PORT} \
    --max-num-seqs ${MAX_BATCH_SIZE} \
    --tensor-parallel-size ${TP_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization 0.9 \
    --mixtral_config_file ${CONFIG_FILE} \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    2>&1 | tee ${STATS_DIR}/${STAT_FILE}
