#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

MODEL=${MODEL:-moonshotai/Kimi-K2.5}
MODEL_TAG=${MODEL_TAG:-kimi_k25}
MODEL_DIR=$(basename "${MODEL}")

# GPQA CoT zeroshot gives Kimi room to reason; set limit to full GPQA by default.
TASK=${TASK:-gpqa_main_cot_zeroshot}
LIMIT=${LIMIT:-448}

PORT=${PORT:-8015}
TP_SIZE=${TP_SIZE:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-16}
NUM_CONCURRENT=${NUM_CONCURRENT:-16}

# "Double output length as usual"
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-2048}

MM_ENCODER_TP_MODE=${MM_ENCODER_TP_MODE:-data}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-kimi_k2}
REASONING_PARSER=${REASONING_PARSER:-kimi_k2}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}

CONFIG_DIR=${CONFIG_DIR:-/nethome/rdudala3/update-vllm/configs/kimi-k2.5}
CONFIGS=(
  "${CONFIG_DIR}/do-nothing.json"
  "${CONFIG_DIR}/quant_alpha1_optimized.json"
)

PERF_STATS_ROOT=${PERF_STATS_ROOT:-"${SCRIPT_DIR}/stats/perf"}
QUALITY_STATS_ROOT=${QUALITY_STATS_ROOT:-"${SCRIPT_DIR}/stats/quality"}

PERF_DIR="${PERF_STATS_ROOT}/spec_decode/ngram/${MODEL_DIR}"
QUALITY_DIR="${QUALITY_STATS_ROOT}/${TASK}/ngram/${MODEL_DIR}"
mkdir -p "${PERF_DIR}" "${QUALITY_DIR}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "Error: 'vllm' not found in PATH."
  exit 1
fi

if ! command -v lm-eval >/dev/null 2>&1; then
  echo "Error: 'lm-eval' not found in PATH."
  exit 1
fi

server_pid=""

cleanup_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    sleep 3
    kill -9 "${server_pid}" 2>/dev/null || true
  fi
  server_pid=""
}

trap cleanup_server EXIT INT TERM

cleanup_port() {
  lsof -ti:"${PORT}" | xargs -r kill -9 2>/dev/null || true
  sleep 2
}

wait_for_server() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
      return 1
    fi
    sleep 5
  done
  return 1
}

echo "Model: ${MODEL}"
echo "Task: ${TASK}"
echo "Limit: ${LIMIT}"
echo "GPUs: ${GPUS} | TP: ${TP_SIZE} | Batch: ${MAX_BATCH_SIZE}"
echo "Max model len: ${MAX_MODEL_LEN} | Max gen toks: ${MAX_GEN_TOKS}"
echo "Configs:"
printf '  - %s\n' "${CONFIGS[@]}"
echo "========================================="

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "Missing config file: ${config}"
    exit 1
  fi

  config_name=$(basename "${config}")
  config_stem="${config_name%.json}"
  stat_prefix="adv_fp16_${MODEL_TAG}_${TASK}_port${PORT}"
  perf_log="${PERF_DIR}/${stat_prefix}_${config_name}_port${PORT}.log"
  quality_stat="${stat_prefix}_n${LIMIT}_conf_${config_stem}"
  quality_log="${QUALITY_DIR}/${quality_stat}.log"
  quality_out="${QUALITY_DIR}/${quality_stat}.jsonl"
  metrics_file="${QUALITY_DIR}/${quality_stat}_$(date +%Y%m%d-%H%M%S).metrics"

  echo "Running config: ${config_name}"
  cleanup_port

  CUDA_VISIBLE_DEVICES="${GPUS}" vllm serve "${MODEL}" \
    --host localhost \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-num-seqs "${MAX_BATCH_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --mm-encoder-tp-mode "${MM_ENCODER_TP_MODE}" \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --reasoning-parser "${REASONING_PARSER}" \
    --mixtral-config-file "${config}" \
    --trust-remote-code \
    > "${perf_log}" 2>&1 &
  server_pid=$!

  if ! wait_for_server; then
    echo "Server failed to become healthy for ${config_name}."
    echo "Last 80 lines from ${perf_log}:"
    tail -n 80 "${perf_log}" || true
    exit 1
  fi

  model_args="model=${MODEL},base_url=http://localhost:${PORT}/v1/chat/completions,add_bos_token=True,max_model_len=${MAX_MODEL_LEN},max_length=${MAX_MODEL_LEN},num_concurrent=${NUM_CONCURRENT}"

  lm-eval \
    --model local-chat-completions \
    --tasks "${TASK}" \
    --model_args "${model_args}" \
    --apply_chat_template \
    --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS},temperature=0.0,do_sample=False" \
    --limit "${LIMIT}" \
    --log_samples \
    --output_path "${quality_out}" \
    2>&1 | tee "${quality_log}"

  curl -fsS "http://localhost:${PORT}/metrics" > "${metrics_file}" || true
  echo "Saved metrics: ${metrics_file}"

  cleanup_server
  sleep 2
  echo "Completed config: ${config_name}"
  echo "-----------------------------------------"
done

echo "All Kimi K2.5 GPQA runs complete."
echo "Perf logs: ${PERF_DIR}"
echo "Quality outputs: ${QUALITY_DIR}"
