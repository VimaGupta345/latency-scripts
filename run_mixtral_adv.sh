#!/bin/bash

export STATFILENAME="adv_sweep_fp16"
export MODEL="/scratch/vgupta345/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0/"

# List of benchmarks
benchmarks=("humaneval" "gsm8k" "mbpp" "minerva_math_algebra")
# benchmarks=("humaneval")

# List of config files
config_files=(
    "/nethome/vgupta345/prowl/configs/mixtral/do_nothing.json"
    "/nethome/vgupta345/prowl/configs/mixtral/advanced_alpha0_beta0.8.json"
    "/nethome/vgupta345/prowl/configs/mixtral/advanced_alpha0_beta1.1.json"
    "/nethome/vgupta345/prowl/configs/mixtral/advanced_alpha0_beta0.9.json"
    "/nethome/vgupta345/prowl/configs/mixtral/advanced_alpha0_beta1.json"
    "/nethome/vgupta345/prowl/configs/mixtral/advanced_alpha0_beta1.25.json"
    "/nethome/vgupta345/prowl/configs/mixtral/advanced_alpha0_beta0.7.json"
)

for cf in "${config_files[@]}"; do
    for benchmark in "${benchmarks[@]}"; do
        echo "Running benchmark: $benchmark with config: $cf"
        python lm_eval_online_serve.py -m "${MODEL}" -o "${STATFILENAME}" -b "${benchmark}" -l 250 -k 0 -t 60 -cf "${cf}"
    done
done

export STATFILENAME="adv_sweep"
export MODEL="/scratch/vgupta345/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0/"

# List of benchmarks
benchmarks=("humaneval" "gsm8k" "mbpp" "minerva_math_algebra")
# benchmarks=("humaneval")

# List of config files
config_files=(
    "/nethome/vgupta345/prowl/configs/mixtral/do_nothing.json"
)

for cf in "${config_files[@]}"; do
    for benchmark in "${benchmarks[@]}"; do
        echo "Running benchmark: $benchmark with config: $cf"
        python lm_eval_online_serve.py -m "${MODEL}" -o "${STATFILENAME}" -b "${benchmark}" -l 250 -k 0 -t 180 -cf "${cf}"
    done
done