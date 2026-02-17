#!/bin/bash

# Check if required parameters are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <port> <model_name> <model_path>"
    echo "Example: $0 8000 mixtral /scratch/shared_dir/models_dir/Mixtral-8x7B-Instruct-v0.1/"
    echo ""
    echo "This script will run multiple benchmarks with different limits and configs"
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

# Define configs to test
config_files=(
    "/nethome/rdudala3/prowl/configs/mixtral/do_nothing.json"
    "/nethome/rdudala3/prowl/configs/mixtral/advanced_alpha0_beta1.25.json"
)

# Define benchmarks with their limits
# Format: "benchmark:limit"
benchmarks=(
    "humaneval:164"
    "gsm8k:100"
)

# Loop through all combinations
for cf in "${config_files[@]}"; do
    config_name=$(basename "$cf" .json)
    echo "========================================="
    echo "Config: $config_name"
    echo "========================================="
    
    for benchmark_spec in "${benchmarks[@]}"; do
        # Split benchmark:limit
        IFS=':' read -r benchmark limit <<< "$benchmark_spec"
        
        echo "Running benchmark: $benchmark (limit: $limit) with config: $cf"
        
        # Run the evaluation
        python lm_eval_online_serve.py \
            -m "${MODEL}" \
            -o "${STATFILENAME}_${config_name}" \
            -b "${benchmark}" \
            -l "${limit}" \
            -k 0 \
            -t 100 \
            -cf "${cf}" \
            -sa "${SERVER_ADDRESS}" \
            -s "./online_serving_ngram_port.sh"
        
        echo "Completed: $benchmark with $config_name"
        echo "-----------------------------------------"
    done
done

echo "All experiments completed on port ${PORT}!"