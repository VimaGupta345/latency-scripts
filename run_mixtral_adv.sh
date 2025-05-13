#!/bin/bash

export STATFILENAME="adv_sweep_fp8"
export MODEL="/home/vimagupta123/.cache/huggingface/hub/models--RedHatAI--Mixtral-8x7B-Instruct-v0.1-FP8/snapshots/9a978d5cc7dabc37b3d13d6501426081c99c4cbd/"

# List of benchmarks
benchmarks=("humaneval" "gsm8k" "mbpp" "minerva_math_algebra")
# benchmarks=("humaneval")

# List of config files
config_files=(
    "/home/vimagupta123/prowl/configs/mixtral/do_nothing.json"
    "/home/vimagupta123/prowl/configs/mixtral/advanced_alpha0_beta0.8.json"
    "/home/vimagupta123/prowl/configs/mixtral/advanced_alpha0_beta1.1.json"
    "/home/vimagupta123/prowl/configs/mixtral/advanced_alpha0_beta0.9.json"
    "/home/vimagupta123/prowl/configs/mixtral/advanced_alpha0_beta1.json"
    "/home/vimagupta123/prowl/configs/mixtral/advanced_alpha0_beta1.25.json"
    "/home/vimagupta123/prowl/configs/mixtral/advanced_alpha0_beta0.7.json"
)

for cf in "${config_files[@]}"; do
    for benchmark in "${benchmarks[@]}"; do
        echo "Running benchmark: $benchmark with config: $cf"
        python lm_eval_online_serve.py -m "${MODEL}" -o "${STATFILENAME}" -b "${benchmark}" -l 250 -k 0 -t 60 -cf "${cf}"
    done
done

export STATFILENAME="adv_sweep"
export MODEL="/home/vimagupta123/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1/"

# List of benchmarks
benchmarks=("humaneval" "gsm8k" "mbpp" "minerva_math_algebra")
# benchmarks=("humaneval")

List of config files
config_files=(
    "/home/vimagupta123/prowl/configs/mixtral/do_nothing.json"
)

for cf in "${config_files[@]}"; do
    for benchmark in "${benchmarks[@]}"; do
        echo "Running benchmark: $benchmark with config: $cf"
        python lm_eval_online_serve.py -m "${MODEL}" -o "${STATFILENAME}" -b "${benchmark}" -l 250 -k 0 -t 180 -cf "${cf}"
    done
done