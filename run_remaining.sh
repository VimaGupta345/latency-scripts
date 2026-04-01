#!/bin/bash
cd "$(dirname "$0")"

# Q3-30B MoE: bs=2,256,512,1024 on GPUs 0,1,2,3 — ports 8050-8053
BATCH_SIZES="2 256 512 1024" NUM_PROMPTS=2048 BASE_PORT=8050 \
./run_max_bs_sweep.sh \
    Qwen/Qwen3-30B-A3B-Instruct-2507 qwen3_30b \
    ../prowl/configs/qwen3_30b/qwen_do-nothing.json 1 "0,1,2,3" &

# Q3-32B dense: bs=1,2 on GPUs 4,5 — ports 8060-8061
BATCH_SIZES="1 2" NUM_PROMPTS=512 BASE_PORT=8060 \
./run_max_bs_sweep.sh \
    Qwen/Qwen3-32B qwen3_32b \
    none 1 "4,5" &

# Q2-57B TP2: bs=1,2 on GPUs 6,7 — ports 8070-8071
BATCH_SIZES="1 2" NUM_PROMPTS=512 BASE_PORT=8070 \
./run_max_bs_sweep.sh \
    Qwen/Qwen2-57B-A14B-Instruct qwen2_57b_tp2 \
    ../prowl/configs/qwen/qwen_do-nothing.json 2 "6,7" &

wait
echo "ALL DONE"
