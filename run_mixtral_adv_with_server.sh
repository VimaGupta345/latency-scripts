#!/bin/bash

# Check if server address is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <server_address> [gpu_id]"
    echo "Example: $0 localhost:8000 0"
    echo "Example: $0 0.0.0.0:8001 1"
    exit 1
fi

SERVER_ADDRESS=$1
GPU_ID=${2:-0}  # Default to GPU 0 if not specified

export STATFILENAME="adv_fp16_mixtral_instruct_gpu${GPU_ID}"
export MODEL="/scratch/vgupta345/models_dir/Mixtral-8x7B-Instruct-v0.1/"
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Running on GPU: $GPU_ID"
echo "Server address: $SERVER_ADDRESS"

# List of benchmarks
benchmarks=("humaneval" "gsm8k" "mbpp" "minerva_math_algebra")

# List of config files
config_files=(
    "/var/tmp/jae/prowl/configs/mixtral/do_nothing.json"
    "/var/tmp/jae/prowl/configs/mixtral/advanced_alpha0_beta1.25.json"
)

for cf in "${config_files[@]}"; do
    for benchmark in "${benchmarks[@]}"; do
        echo "Running benchmark: $benchmark with config: $cf on server: $SERVER_ADDRESS"
        python lm_eval_online_serve.py \
            -m "${MODEL}" \
            -o "${STATFILENAME}" \
            -b "${benchmark}" \
            -k 0 \
            -t 100 \
            -cf "${cf}" \
            -sa "${SERVER_ADDRESS}"
    done
done