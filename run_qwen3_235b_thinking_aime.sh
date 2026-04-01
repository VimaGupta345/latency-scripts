#!/bin/bash

# Qwen3-235B thinking mode on AIME24: baseline vs prowl
# Measures TPOT speedup on long thinking chains

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
source "${TMP_HOME}/prowl/.venv/bin/activate"

GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export VLLM_USE_V1=1
export HF_ALLOW_CODE_EVAL=1

MODEL="Qwen/Qwen3-235B-A22B-Thinking-2507"
OUTDIR="${TMP_HOME}/results/Qwen3-235B-A22B-Thinking-2507/aime24"
mkdir -p "${OUTDIR}"

LIMIT="${1:-10}"

configs=(
    "${TMP_HOME}/prowl/configs/qwen3_235b/qwen_do-nothing.json"
    "${TMP_HOME}/prowl/configs/qwen3_235b/quant_alpha3_beta2_optimized.json"
)

for CONFIG in "${configs[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG" .json)
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    OUTFILE="${OUTDIR}/thinking_${CONFIG_NAME}_${TIMESTAMP}"

    echo "========================================="
    echo "Running AIME24 with ${CONFIG_NAME}"
    echo "  GPUs: ${GPUS}, limit: ${LIMIT}"
    echo "  Output: ${OUTFILE}"
    echo "========================================="

    lm-eval --model vllm \
      --model_args "pretrained=${MODEL},tensor_parallel_size=4,enable_thinking=True,think_end_token=</think>,max_model_len=40960,max_gen_toks=32768,gpu_memory_utilization=0.9,trust_remote_code=True,mixtral_config_file=${CONFIG}" \
      --apply_chat_template \
      --tasks aime24 \
      --limit ${LIMIT} \
      --trust_remote_code --confirm_run_unsafe_code \
      --log_samples --seed 42 \
      --gen_kwargs '{"max_gen_toks": 32768, "temperature": 0.6, "top_p": 0.95, "do_sample": true}' \
      --output_path "${OUTFILE}.jsonl" \
      2>&1 | tee "${OUTFILE}.log"

    echo "Completed: ${CONFIG_NAME}"
    echo ""
done

echo "========================================="
echo "Done. Compare results in ${OUTDIR}/"
echo "========================================="
