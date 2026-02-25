#!/bin/bash

export MODEL=${1}
export STATFILENAME=${2}
export K=${3}
export MAXEXP=${4:-8}
export CONF_THRES=${5:-1.0}
export CONFIG_FILE=${6}
export PORT=${7:-8000}  # New port parameter, default to 8000
export MAX_BATCH_SIZE=${8:-16}

export MODELNAME=$( basename ${MODEL} )
export CONFIGNAME=$( basename ${CONFIG_FILE} )
export STATS_DIR="/nethome/vgupta345/j_prown_opensource/results/server_logs/${MODELNAME}"
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
    --compilation-config '{"full_cuda_graph": true}' \
    --gpu-memory-utilization 0.9  \
    --mixtral_config_file ${CONFIG_FILE} \
    --trust-remote-code 2>&1 | tee ${STATS_DIR}/${STAT_FILE}
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
    --trust-remote-code 2>&1 | tee ${STATS_DIR}/${STAT_FILE}
fi
