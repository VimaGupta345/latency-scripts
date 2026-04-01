#!/bin/bash
#
# Run aiperf trace-replay benchmark against baseline and Prowl configs.
# Supports both standard (single vLLM) and disaggregated (prefill+decode+proxy) modes.
# Supports Mooncake traces (with timestamps) and ShareGPT (rate-based).
#
# Usage:
#   ./run_aiperf_trace_benchmark.sh <model_path> <model_name> <config_dir> [dataset_mode]
#
# dataset_mode:  mooncake (default) | sharegpt
#
# Disaggregated mode (set DISAGG=1):
#   DISAGG=1 PREFILL_GPUS=0,1 DECODE_GPUS=2,3 TP_SIZE=2 \
#   MOONCAKE_TRACE=datasets/mooncake_traces/conversation_trace.jsonl \
#     ./run_aiperf_trace_benchmark.sh Qwen/Qwen2-57B-A14B-Instruct qwen prowl/configs/qwen
#
# Standard mode:
#   CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 \
#   MOONCAKE_TRACE=datasets/mooncake_traces/conversation_trace.jsonl \
#     ./run_aiperf_trace_benchmark.sh Qwen/Qwen2-57B-A14B-Instruct qwen prowl/configs/qwen

set -euo pipefail

export TMP_HOME="${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}"
source "${TMP_HOME}/prowl/.venv/bin/activate"

# ── Arguments ───────────────────────────────────────────────────────
MODEL_PATH="${1:?Usage: $0 <model_path> <model_name> <config_dir> [dataset_mode]}"
MODEL_NAME="${2:?Usage: $0 <model_path> <model_name> <config_dir> [dataset_mode]}"
CONFIG_DIR="${3:?Usage: $0 <model_path> <model_name> <config_dir> [dataset_mode]}"
DATASET_MODE="${4:-mooncake}"

# ── Tunables (override via environment) ─────────────────────────────
TP_SIZE="${TP_SIZE:-2}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

# Dataset
MOONCAKE_TRACE="${MOONCAKE_TRACE:-}"
REQUEST_RATE="${REQUEST_RATE:-4}"
REQUEST_COUNT="${REQUEST_COUNT:-500}"
TRACE_END_OFFSET="${TRACE_END_OFFSET:-300000}"  # first 5 min = 300000 ms

# aiperf
DURATION="${DURATION:-}"
CONCURRENCY="${CONCURRENCY:-}"
STREAMING="${STREAMING:-true}"
TOKENIZER="${TOKENIZER:-${MODEL_PATH}}"
EXPORT_LEVEL="${EXPORT_LEVEL:-records}"

# Disaggregated mode
DISAGG="${DISAGG:-0}"
PREFILL_GPUS="${PREFILL_GPUS:-0,1}"
DECODE_GPUS="${DECODE_GPUS:-2,3}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
PROXY_PORT="${PROXY_PORT:-8000}"
PREFILL_SIDE_CHANNEL="${PREFILL_SIDE_CHANNEL:-5559}"
DECODE_SIDE_CHANNEL="${DECODE_SIDE_CHANNEL:-5659}"

# Standard mode
PORT="${PORT:-8050}"

export VLLM_USE_V1=1

# The port aiperf talks to: proxy in disagg mode, vLLM directly otherwise
if [ "${DISAGG}" = "1" ]; then
    AIPERF_PORT="${PROXY_PORT}"
    MODE_LABEL="disagg"
else
    AIPERF_PORT="${PORT}"
    MODE_LABEL="standard"
fi

RUN_TAG="${RUN_TAG:-}"
if [ -n "${RUN_TAG}" ]; then
    RESULTS_DIR="${TMP_HOME}/results/${MODEL_NAME}_aiperf_${DATASET_MODE}_${MODE_LABEL}_${RUN_TAG}"
else
    RESULTS_DIR="${TMP_HOME}/results/${MODEL_NAME}_aiperf_${DATASET_MODE}_${MODE_LABEL}"
fi
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

PROWL_ROOT="${TMP_HOME}/prowl"
LOGDIR="${RESULTS_DIR}/logs"
mkdir -p "${LOGDIR}"

# ── Helper functions ────────────────────────────────────────────────

kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti:"${port}" 2>/dev/null) || true
    if [ -n "${pids}" ]; then
        for pid in ${pids}; do
            local sid
            sid=$(ps -o sid= -p "${pid}" 2>/dev/null | tr -d ' ') || true
            if [ -n "${sid}" ] && [ "${sid}" != "0" ]; then
                kill -9 -"${sid}" 2>/dev/null || true
            fi
            kill -9 "${pid}" 2>/dev/null || true
        done
        sleep 5
    fi
}

kill_all_ports() {
    if [ "${DISAGG}" = "1" ]; then
        for p in ${PREFILL_PORT} ${DECODE_PORT} ${PROXY_PORT}; do
            kill_port "${p}"
        done
    else
        kill_port "${PORT}"
    fi
    # Kill orphan GPU processes owned by us
    local my_uid
    my_uid=$(id -u)
    for gpu_pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        gpu_pid=$(echo "$gpu_pid" | tr -d ' ')
        local owner
        owner=$(ps -o uid= -p "$gpu_pid" 2>/dev/null | tr -d ' ') || true
        if [ "$owner" = "$my_uid" ]; then
            kill -9 "$gpu_pid" 2>/dev/null || true
        fi
    done
    sleep 3
}

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=600
    local elapsed=0
    echo "    Waiting for ${name} (port ${port})..."
    while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 3
        elapsed=$((elapsed + 3))
        if [ ${elapsed} -ge ${max_wait} ]; then
            echo "    ERROR: ${name} did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "    ${name} ready (${elapsed}s)"
}

# ── Server launch functions ─────────────────────────────────────────

launch_disagg() {
    local config_file=$1
    local config_label=$2

    export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

    local plog="${LOGDIR}/${config_label}_prefill_${TIMESTAMP}.log"
    local dlog="${LOGDIR}/${config_label}_decode_${TIMESTAMP}.log"
    local xlog="${LOGDIR}/${config_label}_proxy_${TIMESTAMP}.log"

    # Launch prefill
    setsid bash -c "
        CUDA_VISIBLE_DEVICES=${PREFILL_GPUS} \
        VLLM_NIXL_SIDE_CHANNEL_PORT=${PREFILL_SIDE_CHANNEL} \
        python -m vllm.entrypoints.openai.api_server \
            --model ${MODEL_PATH} \
            --host 0.0.0.0 \
            --port ${PREFILL_PORT} \
            --tensor-parallel-size ${TP_SIZE} \
            --max-model-len ${MAX_MODEL_LEN} \
            --max-num-seqs ${MAX_BATCH_SIZE} \
            --gpu-memory-utilization ${GPU_MEM_UTIL} \
            --mixtral_config_file ${config_file} \
            --trust-remote-code \
            --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' \
        > '${plog}' 2>&1
    " &
    PIDS+=($!)

    # Launch decode
    setsid bash -c "
        CUDA_VISIBLE_DEVICES=${DECODE_GPUS} \
        VLLM_NIXL_SIDE_CHANNEL_PORT=${DECODE_SIDE_CHANNEL} \
        python -m vllm.entrypoints.openai.api_server \
            --model ${MODEL_PATH} \
            --host 0.0.0.0 \
            --port ${DECODE_PORT} \
            --tensor-parallel-size ${TP_SIZE} \
            --max-model-len ${MAX_MODEL_LEN} \
            --max-num-seqs ${MAX_BATCH_SIZE} \
            --gpu-memory-utilization ${GPU_MEM_UTIL} \
            --mixtral_config_file ${config_file} \
            --trust-remote-code \
            --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' \
        > '${dlog}' 2>&1
    " &
    PIDS+=($!)

    # Wait for both
    if ! wait_for_server ${PREFILL_PORT} "prefill"; then return 1; fi
    if ! wait_for_server ${DECODE_PORT} "decode"; then return 1; fi

    # Launch proxy
    setsid python3 "${PROWL_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
        --port "${PROXY_PORT}" \
        --prefiller-hosts localhost --prefiller-ports "${PREFILL_PORT}" \
        --decoder-hosts localhost --decoder-ports "${DECODE_PORT}" \
        > "${xlog}" 2>&1 &
    PIDS+=($!)
    sleep 3
    echo "    Proxy ready on port ${PROXY_PORT}"
}

launch_standard() {
    local config_file=$1
    local config_label=$2
    local server_log="${LOGDIR}/${config_label}_server_${TIMESTAMP}.log"

    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --host localhost \
        --port "${PORT}" \
        --max-num-seqs "${MAX_BATCH_SIZE}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --mixtral_config_file "${config_file}" \
        --trust-remote-code \
        2>&1 | tee "${server_log}" &
    PIDS+=($!)

    if ! wait_for_server "${PORT}" "vllm"; then return 1; fi
}

# ── aiperf invocation ──────────────────────────────────────────────

run_aiperf() {
    local label=$1
    local artifact_dir=$2

    local cmd=(
        aiperf profile
        --model "${MODEL_PATH}"
        --url "http://localhost:${AIPERF_PORT}"
        --endpoint-type chat
        --tokenizer "${TOKENIZER}"
        --output-artifact-dir "${artifact_dir}"
        --profile-export-prefix "${label}"
        --export-level "${EXPORT_LEVEL}"
        --use-legacy-max-tokens
    )

    if [ "${STREAMING}" = "true" ]; then
        cmd+=(--streaming)
    fi

    if [ -n "${CONCURRENCY}" ]; then
        cmd+=(--concurrency "${CONCURRENCY}")
    fi

    if [ -n "${DURATION}" ]; then
        cmd+=(--duration "${DURATION}")
    fi

    case "${DATASET_MODE}" in
        mooncake)
            if [ -z "${MOONCAKE_TRACE}" ]; then
                echo "  ERROR: MOONCAKE_TRACE must be set for mooncake mode"
                return 1
            fi
            cmd+=(
                --input-file "${MOONCAKE_TRACE}"
                --custom-dataset-type mooncake-trace
                --fixed-schedule
                --fixed-schedule-auto-offset
                --fixed-schedule-end-offset "${TRACE_END_OFFSET}"
            )
            ;;
        sharegpt)
            cmd+=(
                --public-dataset sharegpt
                --request-rate "${REQUEST_RATE}"
                --request-count "${REQUEST_COUNT}"
            )
            ;;
        *)
            echo "  ERROR: Unknown dataset_mode '${DATASET_MODE}'."
            return 1
            ;;
    esac

    echo "  Running: ${cmd[*]}"
    "${cmd[@]}" 2>&1 | tee "${artifact_dir}/${label}_console.log"
}

# ── Run one config ──────────────────────────────────────────────────

run_one_config() {
    local config_file=$1
    local config_label=$2

    PIDS=()

    echo ""
    echo "============================================================"
    echo "  Config:   ${config_label} ($(basename "${config_file}"))"
    echo "  Model:    ${MODEL_PATH}"
    echo "  Mode:     ${MODE_LABEL} (TP=${TP_SIZE})"
    echo "  Dataset:  ${DATASET_MODE} (end_offset=${TRACE_END_OFFSET}ms)"
    echo "  Time:     ${TIMESTAMP}"
    echo "============================================================"

    kill_all_ports

    # Launch servers
    if [ "${DISAGG}" = "1" ]; then
        if ! launch_disagg "${config_file}" "${config_label}"; then
            echo "  ERROR: Failed to launch disagg servers. Skipping."
            kill_all_ports
            return 1
        fi
    else
        if ! launch_standard "${config_file}" "${config_label}"; then
            echo "  ERROR: Failed to launch vLLM. Skipping."
            kill_all_ports
            return 1
        fi
    fi

    # Run aiperf
    local artifact_dir="${RESULTS_DIR}/${config_label}_${TIMESTAMP}"
    mkdir -p "${artifact_dir}"
    run_aiperf "${config_label}" "${artifact_dir}" || true

    # Collect Prometheus metrics
    if [ "${DISAGG}" = "1" ]; then
        curl -s "http://localhost:${PREFILL_PORT}/metrics" > \
            "${artifact_dir}/${config_label}_prefill.metrics" 2>/dev/null || true
        curl -s "http://localhost:${DECODE_PORT}/metrics" > \
            "${artifact_dir}/${config_label}_decode.metrics" 2>/dev/null || true
    else
        curl -s "http://localhost:${PORT}/metrics" > \
            "${artifact_dir}/${config_label}_final.metrics" 2>/dev/null || true
    fi

    # Tear down
    echo "    Tearing down servers..."
    for pid in "${PIDS[@]}"; do
        local sid
        sid=$(ps -o sid= -p "${pid}" 2>/dev/null | tr -d ' ') || true
        if [ -n "${sid}" ] && [ "${sid}" != "0" ]; then
            kill -9 -"${sid}" 2>/dev/null || true
        fi
        kill -9 "${pid}" 2>/dev/null || true
    done
    kill_all_ports

    echo "  Done: ${config_label} -> ${artifact_dir}"
}

# ── Discover configs ────────────────────────────────────────────────

find_config() {
    local dir=$1
    local pattern=$2
    find "${dir}" -maxdepth 1 -name "${pattern}" -type f 2>/dev/null | head -1
}

BASELINE_CONFIG="${BASELINE_CONFIG:-$(find_config "${CONFIG_DIR}" "*do?nothing*")}"
PROWL_CONFIG="${PROWL_CONFIG:-$(find_config "${CONFIG_DIR}" "*quant_*optimized*")}"

if [ -z "${BASELINE_CONFIG}" ] || [ -z "${PROWL_CONFIG}" ]; then
    echo "Available configs in ${CONFIG_DIR}:"
    ls -1 "${CONFIG_DIR}"/ 2>/dev/null || echo "  (none)"
    echo ""
    [ -z "${BASELINE_CONFIG}" ] && echo "ERROR: Set BASELINE_CONFIG=..." && exit 1
    [ -z "${PROWL_CONFIG}" ] && echo "ERROR: Set PROWL_CONFIG=..." && exit 1
fi

# ── Main ────────────────────────────────────────────────────────────

echo "=================================================================="
echo "  AIPerf Trace Benchmark (${MODE_LABEL})"
echo "  Model:      ${MODEL_PATH}"
echo "  Dataset:    ${DATASET_MODE}"
echo "  Trace end:  ${TRACE_END_OFFSET}ms"
echo "  TP:         ${TP_SIZE}"
if [ "${DISAGG}" = "1" ]; then
echo "  Prefill:    GPUs ${PREFILL_GPUS}, port ${PREFILL_PORT}"
echo "  Decode:     GPUs ${DECODE_GPUS}, port ${DECODE_PORT}"
echo "  Proxy:      port ${PROXY_PORT}"
else
echo "  Port:       ${PORT}"
fi
echo "  Baseline:   $(basename "${BASELINE_CONFIG}")"
echo "  Prowl:      $(basename "${PROWL_CONFIG}")"
echo "  Results:    ${RESULTS_DIR}"
echo "=================================================================="

SKIP_BASELINE="${SKIP_BASELINE:-0}"

configs=()
if [ "${SKIP_BASELINE}" != "1" ]; then
    configs+=("${BASELINE_CONFIG}:baseline")
fi
configs+=("${PROWL_CONFIG}:prowl")

for entry in "${configs[@]}"; do
    config_file="${entry%%:*}"
    config_label="${entry##*:}"
    run_one_config "${config_file}" "${config_label}"
done

# ── Compare results ─────────────────────────────────────────────────

echo ""
echo "Generating comparison..."

python3 - "${RESULTS_DIR}" "${TIMESTAMP}" <<'PYEOF'
import sys, os, json, glob

results_dir = sys.argv[1]
timestamp = sys.argv[2]

def load_aiperf_json(artifact_dir):
    """Load the aiperf summary JSON from an artifact directory."""
    candidates = glob.glob(f"{artifact_dir}/*_aiperf.json") + \
                 glob.glob(f"{artifact_dir}/*.json")
    for c in candidates:
        if "_console" in c:
            continue
        try:
            with open(c) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data, c
        except (json.JSONDecodeError, KeyError):
            continue
    return None, None

def extract_metrics(data):
    """Extract key metrics from aiperf output, handling nested structures."""
    if data is None:
        return {}
    metrics = {}
    src = data.get("metrics", data)
    key_map = {
        "time_to_first_token_avg": "mean_ttft_ms",
        "time_to_first_token_p50": "median_ttft_ms",
        "time_to_first_token_p99": "p99_ttft_ms",
        "inter_token_latency_avg": "mean_itl_ms",
        "inter_token_latency_p50": "median_itl_ms",
        "inter_token_latency_p99": "p99_itl_ms",
        "output_token_throughput": "output_throughput",
        "request_throughput": "request_throughput",
        "request_latency_avg": "mean_e2el_ms",
        "request_latency_p50": "median_e2el_ms",
        "request_latency_p99": "p99_e2el_ms",
    }
    for src_key, dst_key in key_map.items():
        if src_key in src:
            metrics[dst_key] = float(src[src_key])
    if not metrics:
        for key, val in src.items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    flat_key = f"{key}_{sub_key}"
                    if flat_key in key_map:
                        metrics[key_map[flat_key]] = float(sub_val)
                    metrics[flat_key] = float(sub_val) if isinstance(sub_val, (int, float)) else sub_val
    if not metrics:
        for k, v in src.items():
            if isinstance(v, (int, float)):
                metrics[k] = v
    return metrics

baseline_dir = glob.glob(f"{results_dir}/baseline_{timestamp}")
prowl_dir = glob.glob(f"{results_dir}/prowl_{timestamp}")

results = {}
for label, dirs in [("baseline", baseline_dir), ("prowl", prowl_dir)]:
    if not dirs or not os.path.isdir(dirs[0]):
        print(f"  {label}: no results directory found")
        continue
    data, path = load_aiperf_json(dirs[0])
    if data is None:
        print(f"  {label}: no results JSON found in {dirs[0]}")
        continue
    metrics = extract_metrics(data)
    results[label] = metrics
    print(f"\n  {label} ({os.path.basename(path)}):")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, float):
            print(f"    {k:35s} {v:.3f}")
        else:
            print(f"    {k:35s} {v}")

if "baseline" in results and "prowl" in results:
    b = results["baseline"]
    p = results["prowl"]
    print(f"\n  {'─' * 60}")
    print(f"  PROWL Improvement:")
    comparisons = [
        ("median_ttft_ms",    "TTFT (p50)",       "latency"),
        ("p99_ttft_ms",       "TTFT (p99)",       "latency"),
        ("mean_ttft_ms",      "TTFT (mean)",      "latency"),
        ("median_itl_ms",     "ITL (p50)",        "latency"),
        ("p99_itl_ms",        "ITL (p99)",        "latency"),
        ("mean_itl_ms",       "ITL (mean)",       "latency"),
        ("median_e2el_ms",    "E2E Latency (p50)","latency"),
        ("p99_e2el_ms",       "E2E Latency (p99)","latency"),
        ("output_throughput", "Output tok/s",     "throughput"),
        ("request_throughput","Request/s",        "throughput"),
    ]
    for key, label, kind in comparisons:
        bv = b.get(key, 0)
        pv = p.get(key, 0)
        if bv and pv:
            if kind == "throughput":
                ratio = pv / bv
            else:
                ratio = bv / pv
            marker = "+" if ratio > 1.0 else ""
            pct = (ratio - 1.0) * 100
            print(f"    {label:25s} {ratio:.3f}x ({marker}{pct:.1f}%)  "
                  f"[base={bv:.2f}, prowl={pv:.2f}]")

    comparison_file = f"{results_dir}/comparison_{timestamp}.json"
    with open(comparison_file, "w") as f:
        json.dump({"timestamp": timestamp, "baseline": b, "prowl": p}, f, indent=2)
    print(f"\n  Comparison JSON: {comparison_file}")

print(f"\n  All results in: {results_dir}")
print("=" * 70)
PYEOF

echo ""
echo "Done! Results in: ${RESULTS_DIR}"
