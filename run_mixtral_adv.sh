#!/bin/bash

export STATFILENAME="adv_fp16_mixtral_instruct"
export MODEL="/scratch/vgupta345/models_dir/Mixtral-8x7B-Instruct-v0.1/"

# List of benchmarks
benchmarks=("humaneval" "gsm8k" "mbpp" "minerva_math_algebra")
# benchmarks=("humaneval")

# List of config files
config_files=(
    "/nethome/vgupta345/prowl/configs/mixtral/do_nothing.json"
    "/nethome/vgupta345/prowl/configs/mixtral/advanced_alpha0_beta1.25.json"
)

for cf in "${config_files[@]}"; do
    for benchmark in "${benchmarks[@]}"; do
        echo "Running benchmark: $benchmark with config: $cf"
        python lm_eval_online_serve.py -m "${MODEL}" -o "${STATFILENAME}" -b "${benchmark}" -k 0 -t 100 -cf "${cf}"
    done
done

# export STATFILENAME="adv_sweep"
# export MODEL="/scratch/vgupta345/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0/"

# # List of benchmarks
# benchmarks=("humaneval" "gsm8k" "mbpp" "minerva_math_algebra")
# # benchmarks=("humaneval")

# # List of config files
# config_files=(
#     "/nethome/vgupta345/prowl/configs/mixtral/do_nothing.json"
# )

# for cf in "${config_files[@]}"; do
#     for benchmark in "${benchmarks[@]}"; do
#         echo "Running benchmark: $benchmark with config: $cf"
#         python lm_eval_online_serve.py -m "${MODEL}" -o "${STATFILENAME}" -b "${benchmark}" -l 250 -k 0 -t 180 -cf "${cf}"
#     done
# done