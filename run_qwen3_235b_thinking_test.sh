#!/bin/bash

# Quick test: Qwen3-235B thinking mode on humaneval via in-process vLLM
# Uses think_end_token to strip thinking content before evaluation

export TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source
source "${TMP_HOME}/prowl/.venv/bin/activate"

GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export VLLM_USE_V1=1
export HF_ALLOW_CODE_EVAL=1

MODEL="Qwen/Qwen3-235B-A22B-Thinking-2507"
CONFIG="${TMP_HOME}/prowl/configs/qwen3_235b/qwen_do-nothing.json"
OUTDIR="${TMP_HOME}/results/Qwen3-235B-A22B-Thinking-2507/humaneval"
mkdir -p "${OUTDIR}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTFILE="${OUTDIR}/thinking_test_baseline_${TIMESTAMP}"

echo "Running Qwen3-235B humaneval with thinking mode (baseline)"
echo "  GPUs: ${GPUS}"
echo "  max_gen_toks: 2048 (thinking + answer)"
echo "  Output: ${OUTFILE}"

lm-eval --model vllm \
  --model_args "pretrained=${MODEL},tensor_parallel_size=4,enable_thinking=True,think_end_token=</think>,max_model_len=40960,max_gen_toks=32768,gpu_memory_utilization=0.9,trust_remote_code=True,mixtral_config_file=${CONFIG}" \
  --apply_chat_template \
  --tasks aime24 \
  --limit 5 \
  --trust_remote_code --confirm_run_unsafe_code \
  --log_samples --seed 42 \
  --gen_kwargs '{"max_gen_toks": 32768, "temperature": 0.6, "top_p": 0.95, "do_sample": true}' \
  --output_path "${OUTFILE}.jsonl" \
  2>&1 | tee "${OUTFILE}.log"

echo "Done. Check ${OUTFILE}.log"
