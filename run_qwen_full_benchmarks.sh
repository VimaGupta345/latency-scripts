#!/bin/bash

# Qwen Full Benchmarks - 99% completion
# Runs on GPUs 2,3 with TP=2 on port 8001

MODEL_PATH="Qwen/Qwen2-57B-A14B-Instruct"
MODEL_NAME="qwen"
PORT=8009
TP_SIZE=2
GPUS="0, 1"
ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL:-false}
ENABLE_DISAGG_SERVING=${ENABLE_DISAGG_SERVING:-false}

GPUS_CLEAN=${GPUS// /}
IFS=',' read -ra GPU_LIST <<< "${GPUS_CLEAN}"
DEFAULT_PREFILL_GPU=${GPU_LIST[0]:-0}
DEFAULT_DECODE_GPU=${GPU_LIST[1]:-${GPU_LIST[0]:-0}}

DISAGG_PROXY_PORT=${DISAGG_PROXY_PORT:-${PORT}}
DISAGG_PREFILL_PORT=${DISAGG_PREFILL_PORT:-$((PORT+1))}
DISAGG_DECODE_PORT=${DISAGG_DECODE_PORT:-$((PORT+2))}
DISAGG_PREFILL_GPUS=${DISAGG_PREFILL_GPUS:-${DEFAULT_PREFILL_GPU}}
DISAGG_DECODE_GPUS=${DISAGG_DECODE_GPUS:-${DEFAULT_DECODE_GPU}}
DISAGG_KV_PORT=${DISAGG_KV_PORT:-14579}
DISAGG_KV_CONNECTOR=${DISAGG_KV_CONNECTOR:-"PyNcclConnector"}

# Config files for Qwen
configs=(
    "${TMP_HOME}/prowl/configs/qwen/quant_alpha1_optimized.json"
    "${TMP_HOME}/prowl/configs/qwen/qwen_do-nothing.json"
)

# Use 99% completion for all benchmarks
LIMIT_PERCENT=0.99

# List of benchmarks to run
# Using default limits from lm_eval_online_serve.py
benchmarks=("gsm8k")
#("gsm8k" "mbpp" "minerva_math_algebra" "humaneval" "hotpotqa" "xsum")

echo "Starting Qwen full benchmark suite on GPUs ${GPUS}"
echo "Using default limits from lm_eval_online_serve.py"
echo "Expert parallelism: ${ENABLE_EXPERT_PARALLEL}"
echo "Disaggregated serving: ${ENABLE_DISAGG_SERVING}"
if [ "${ENABLE_DISAGG_SERVING}" = "true" ]; then
    echo "  Proxy port: ${DISAGG_PROXY_PORT}, Prefill port: ${DISAGG_PREFILL_PORT}, Decode port: ${DISAGG_DECODE_PORT}"
    echo "  Prefill GPUs: ${DISAGG_PREFILL_GPUS}, Decode GPUs: ${DISAGG_DECODE_GPUS}, KV port: ${DISAGG_KV_PORT}, Connector: ${DISAGG_KV_CONNECTOR}"
fi
echo "========================================="

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .json)
    echo "Config: $config_name"
    echo "-----------------------------------------"
    
    for benchmark in "${benchmarks[@]}"; do
        
        echo "Running $benchmark with $config_name"
        
        # Note: run_mixtral_adv_configurable.sh is a generic runner that works for both Mixtral and Qwen
        # Pass empty string for limit to use defaults
        ENABLE_DISAGG_SERVING=${ENABLE_DISAGG_SERVING} \
        DISAGG_PROXY_PORT=${DISAGG_PROXY_PORT} \
        DISAGG_PREFILL_PORT=${DISAGG_PREFILL_PORT} \
        DISAGG_DECODE_PORT=${DISAGG_DECODE_PORT} \
        DISAGG_KV_PORT=${DISAGG_KV_PORT} \
        DISAGG_PREFILL_GPUS="${DISAGG_PREFILL_GPUS}" \
        DISAGG_DECODE_GPUS="${DISAGG_DECODE_GPUS}" \
        DISAGG_KV_CONNECTOR="${DISAGG_KV_CONNECTOR}" \
        CUDA_VISIBLE_DEVICES=${GPUS} ./run_mixtral_adv_configurable.sh \
            ${PORT} \
            ${MODEL_NAME} \
            ${MODEL_PATH} \
            ${benchmark} \
            "" \
            ${config} \
            ${TP_SIZE} \
            16 \
            ${ENABLE_EXPERT_PARALLEL}
        
        echo "Completed: $benchmark with $config_name"
        echo ""
    done
    echo "========================================="
done

echo "All Qwen benchmarks completed!"
