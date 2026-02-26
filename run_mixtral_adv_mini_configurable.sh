#!/bin/bash

# Check if required parameters are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <port> <model_name> <model_path>"
    echo "Example: $0 8000 mixtral /scratch/vgupta345/models_dir/Mixtral-8x7B-Instruct-v0.1/"
    echo "Example: $0 8001 llama /path/to/llama/model/"
    exit 1
fi

PORT=$1
MODEL_NAME=$2
MODEL_PATH=$3
SERVER_ADDRESS="localhost:${PORT}"

export STATFILENAME="adv_fp16_${MODEL_NAME}_port${PORT}"
export MODEL="${MODEL_PATH}"

# Kill any existing process on this port
echo "Cleaning up port ${PORT}..."
lsof -ti:${PORT} | xargs -r kill -9 2>/dev/null
sleep 2

echo "Using server at: $SERVER_ADDRESS"

# List of benchmarks
benchmarks=("gsm8k")

# List of config files
config_files=(
    "/nethome/jkim3934/prowl/configs/mixtral/do_nothing.json"
    "/nethome/jkim3934/prowl/configs/mixtral/advanced_alpha0_beta0.7.json"
    "/nethome/jkim3934/prowl/configs/mixtral/advanced_alpha0_beta1.25.json"
)

for cf in "${config_files[@]}"; do
    for benchmark in "${benchmarks[@]}"; do
        echo "Running benchmark: $benchmark with config: $cf"
        python lm_eval_online_serve.py \
            -m "${MODEL}" \
            -o "${STATFILENAME}" \
            -b "${benchmark}" \
            -k 0 \
            -t 100 \
            -cf "${cf}" \
            -sa "${SERVER_ADDRESS}" \
            -s "./online_serving_ngram_port.sh"
    done
done