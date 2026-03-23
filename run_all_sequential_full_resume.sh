#!/usr/bin/env zsh

# Resume sequential run — uses default limits from lm_eval_online_serve.py
# gsm8k=900, minerva=900, humaneval=164, mbpp=500 (was 250)
# Skips DeepSeek humaneval+mbpp baseline (already done)
# Usage: zsh run_all_sequential_full_resume.sh

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source

LOGDIR="${TMP_HOME}/results/logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/all_sequential_full_resume_$(date +%Y%m%d-%H%M%S).log"

exec > >(tee -a "${LOGFILE}") 2>&1

echo "========================================="
echo "SEQUENTIAL RUN (RESUMED) — default limits"
echo "Skipping: DeepSeek humaneval+mbpp baseline (already done)"
echo "Log: ${LOGFILE}"
echo "Started: $(date)"
echo "========================================="

TOTAL_RUNS=30
RUN_NUM=0

run_benchmarks() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    local PORT=$3
    local TP=$4
    local GPUS=$5
    local CONFIG=$6
    shift 6
    local BENCHMARKS=("$@")

    export TP_SIZE="${TP}"
    export CUDA_VISIBLE_DEVICES="${GPUS}"

    local config_name=$(basename "$CONFIG" .json)

    for benchmark in ${BENCHMARKS[@]}; do
        RUN_NUM=$((RUN_NUM + 1))
        echo ""
        echo "  [${RUN_NUM}/${TOTAL_RUNS}] ${MODEL_NAME} / ${benchmark} / ${config_name}"
        echo "  $(date)"

        ./run_mixtral_adv_configurable.sh \
            ${PORT} \
            ${MODEL_NAME} \
            ${MODEL_PATH} \
            ${benchmark} \
            "" \
            ${CONFIG} \
            ${TP}

        echo "  Done: ${MODEL_NAME} / ${benchmark} / ${config_name}"
    done
}

# =========================================
# DeepSeek-Coder-V2: TP=4, GPUs 0,1,2,3
# =========================================
echo ""
echo "========================================="
echo "MODEL: deepseek_v2 (deepseek-ai/DeepSeek-Coder-V2-Instruct)"
echo "  Port: 8020, TP: 4, GPUs: 0,1,2,3"
echo "========================================="

# Baseline: only gsm8k + minerva (humaneval+mbpp already done)
run_benchmarks deepseek_v2 \
    deepseek-ai/DeepSeek-Coder-V2-Instruct \
    8020 4 "0,1,2,3" \
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/config_do_nothing.json" \
    gsm8k minerva_math_algebra

# Prowl: all 4
run_benchmarks deepseek_v2 \
    deepseek-ai/DeepSeek-Coder-V2-Instruct \
    8020 4 "0,1,2,3" \
    "${TMP_HOME}/prowl/configs/deepseek_v2_coder/quant_alpha1.125_beta2_optimized.json" \
    humaneval mbpp gsm8k minerva_math_algebra

# =========================================
# Mixtral-8x7B: TP=2, GPUs 0,1
# =========================================
echo ""
echo "========================================="
echo "MODEL: mixtral (mistralai/Mixtral-8x7B-Instruct-v0.1)"
echo "  Port: 8040, TP: 2, GPUs: 0,1"
echo "========================================="

run_benchmarks mixtral \
    mistralai/Mixtral-8x7B-Instruct-v0.1 \
    8040 2 "0,1" \
    "${TMP_HOME}/prowl/configs/mixtral/do_nothing.json" \
    humaneval mbpp gsm8k minerva_math_algebra

run_benchmarks mixtral \
    mistralai/Mixtral-8x7B-Instruct-v0.1 \
    8040 2 "0,1" \
    "${TMP_HOME}/prowl/configs/mixtral/quant_alpha0.7_beta1_optimized.json" \
    humaneval mbpp gsm8k minerva_math_algebra

# =========================================
# Qwen2-57B-A14B: TP=2, GPUs 0,1
# =========================================
echo ""
echo "========================================="
echo "MODEL: qwen (Qwen/Qwen2-57B-A14B-Instruct)"
echo "  Port: 8030, TP: 2, GPUs: 0,1"
echo "========================================="

run_benchmarks qwen \
    Qwen/Qwen2-57B-A14B-Instruct \
    8030 2 "0,1" \
    "${TMP_HOME}/prowl/configs/qwen/qwen_do-nothing.json" \
    humaneval mbpp gsm8k minerva_math_algebra

run_benchmarks qwen \
    Qwen/Qwen2-57B-A14B-Instruct \
    8030 2 "0,1" \
    "${TMP_HOME}/prowl/configs/qwen/quant_alpha3_beta4_optimized.json" \
    humaneval mbpp gsm8k minerva_math_algebra

# =========================================
# Qwen3-30B-A3B: TP=1, GPU 0
# =========================================
echo ""
echo "========================================="
echo "MODEL: qwen3 (Qwen/Qwen3-30B-A3B-Instruct-2507)"
echo "  Port: 8019, TP: 1, GPUs: 0"
echo "========================================="

run_benchmarks qwen3 \
    Qwen/Qwen3-30B-A3B-Instruct-2507 \
    8019 1 "0" \
    "${TMP_HOME}/prowl/configs/qwen3_30b/qwen_do-nothing.json" \
    humaneval mbpp gsm8k minerva_math_algebra

run_benchmarks qwen3 \
    Qwen/Qwen3-30B-A3B-Instruct-2507 \
    8019 1 "0" \
    "${TMP_HOME}/prowl/configs/qwen3_30b/quant_alpha3_beta2_optimized.json" \
    humaneval mbpp gsm8k minerva_math_algebra

echo ""
echo "========================================="
echo "ALL SEQUENTIAL RUNS COMPLETED"
echo "Finished: $(date)"
echo "Total runs: ${TOTAL_RUNS}"
echo "========================================="
