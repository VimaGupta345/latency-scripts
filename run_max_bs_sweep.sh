#!/bin/bash
#
# Sweep --max-num-seqs (server-side batch size limit) and measure TPOT latency.
# Runs ALL batch sizes in PARALLEL — one server per GPU, each on a different port.
#
# Usage:
#   ./run_max_bs_sweep.sh [model_path] [model_name] [config_file] [tp_size] [gpus]
#
# Examples:
#   # Qwen3-30B baseline, TP=1, 8 GPUs in parallel
#   ./run_max_bs_sweep.sh \
#       Qwen/Qwen3-30B-A3B-Instruct-2507 qwen3_30b \
#       ../prowl/configs/qwen3_30b/qwen_do-nothing.json 1 "0,1,2,3,4,5,6,7"
#
#   # Qwen3-235B, TP=4, 2 groups of 4 GPUs (runs 2 batch sizes at a time)
#   ./run_max_bs_sweep.sh \
#       Qwen/Qwen3-235B-A22B-Thinking-2507 qwen3_235b \
#       ../prowl/configs/qwen3_235b/qwen_do-nothing.json 4 "0,1,2,3,4,5,6,7"

set -euo pipefail

export TMP_HOME="${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}"
source "${TMP_HOME}/prowl/.venv/bin/activate"

# ── Configurable parameters ──────────────────────────────────────────
MODEL_PATH="${1:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MODEL_NAME="${2:-qwen3_30b}"
CONFIG_FILE="${3:-${TMP_HOME}/prowl/configs/qwen3_30b/qwen_do-nothing.json}"
TP_SIZE="${4:-1}"
GPUS="${5:-0,1,2,3,4,5,7}"

# Batch sizes to sweep (server-side --max-num-seqs)
BATCH_SIZES=(${BATCH_SIZES:-1 4 8 16 32 64 128})

# Number of concurrent prompts to send (should exceed largest batch size)
NUM_PROMPTS="${NUM_PROMPTS:-512}"
REQUEST_RATE="${REQUEST_RATE:-10000.0}"  # effectively unlimited (all at once)

DATASET="${DATASET:-${TMP_HOME}/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
BASE_PORT="${BASE_PORT:-8050}"

CONFIG_LABEL=$(basename "${CONFIG_FILE}" .json)
RESULTS_DIR="${TMP_HOME}/results/${MODEL_NAME}_bs_sweep"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="${RESULTS_DIR}/${CONFIG_LABEL}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

# ── Parse GPUs into groups of TP_SIZE ────────────────────────────────
IFS=',' read -ra ALL_GPUS <<< "${GPUS}"
NUM_GPUS=${#ALL_GPUS[@]}
SLOTS=$((NUM_GPUS / TP_SIZE))

if [ ${SLOTS} -lt 1 ]; then
    echo "ERROR: Need at least ${TP_SIZE} GPUs for TP=${TP_SIZE}, got ${NUM_GPUS}"
    exit 1
fi

# Build GPU groups (e.g., TP=2: "0,1" "2,3" "4,5" "6,7")
GPU_GROUPS=()
for ((i=0; i<SLOTS; i++)); do
    group=""
    for ((j=0; j<TP_SIZE; j++)); do
        idx=$((i * TP_SIZE + j))
        [ -n "${group}" ] && group+=","
        group+="${ALL_GPUS[$idx]}"
    done
    GPU_GROUPS+=("${group}")
done

echo "============================================================"
echo "  Max Batch Size Sweep (PARALLEL)"
echo "  Model:       ${MODEL_PATH}"
echo "  Config:      ${CONFIG_LABEL}"
echo "  TP:          ${TP_SIZE}"
echo "  GPUs:        ${GPUS} (${SLOTS} parallel slots)"
echo "  Batch sizes: ${BATCH_SIZES[*]}"
echo "  Num prompts: ${NUM_PROMPTS}"
echo "  Results:     ${RUN_DIR}"
echo "============================================================"

# ── Helper functions ─────────────────────────────────────────────────

kill_port() {
    local port=$1
    local pids=$(lsof -ti:${port} 2>/dev/null || true)
    if [ -n "${pids}" ]; then
        for pid in ${pids}; do
            # Only kill the specific PID, NOT the process group (which would nuke sibling jobs)
            kill -9 ${pid} 2>/dev/null || true
        done
        sleep 3
    fi
}

wait_for_server() {
    local port=$1
    local max_wait=600
    local elapsed=0
    while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 3
        elapsed=$((elapsed + 3))
        if [ ${elapsed} -ge ${max_wait} ]; then
            echo "[port ${port}] ERROR: Server did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "[port ${port}] Server ready (${elapsed}s)"
}

# ── Single batch-size run (called as background job) ─────────────────

run_one_bs() {
    local bs=$1
    local gpu_group=$2
    local port=$3

    local bs_dir="${RUN_DIR}/bs_${bs}"
    mkdir -p "${bs_dir}"

    echo "[bs=${bs}] Starting on GPU ${gpu_group}, port ${port}"

    # Kill any leftover on this port
    kill_port ${port}

    # Start vLLM server
    local config_flag=""
    if [ -n "${CONFIG_FILE}" ] && [ "${CONFIG_FILE}" != "none" ]; then
        config_flag="--mixtral-config-file ${CONFIG_FILE}"
    fi
    CUDA_VISIBLE_DEVICES=${gpu_group} python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --host localhost \
        --port ${port} \
        --max-num-seqs ${bs} \
        --tensor-parallel-size ${TP_SIZE} \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        ${config_flag} \
        --trust-remote-code \
        > "${bs_dir}/server.log" 2>&1 &
    local server_pid=$!

    wait_for_server ${port}
    if [ $? -ne 0 ]; then
        echo "[bs=${bs}] FAILED to start server"
        kill ${server_pid} 2>/dev/null || true
        return 1
    fi

    # Run benchmark
    echo "[bs=${bs}] Benchmarking (${NUM_PROMPTS} prompts)..."
    vllm bench serve \
        --backend vllm \
        --model "${MODEL_PATH}" \
        --host 127.0.0.1 \
        --port ${port} \
        --endpoint /v1/completions \
        --dataset-name sharegpt \
        --dataset-path "${DATASET}" \
        --num-prompts ${NUM_PROMPTS} \
        --request-rate ${REQUEST_RATE} \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl \
        --metric-percentiles "50,90,99" \
        --save-result \
        --save-detailed \
        --result-dir "${bs_dir}" \
        --result-filename "bench_bs${bs}.json" \
        > "${bs_dir}/bench.log" 2>&1

    # Scrape Prometheus metrics
    curl -s "http://localhost:${port}/metrics" > "${bs_dir}/server.metrics"

    # Parse metrics
    python "${TMP_HOME}/latency-scripts/get_vllm_metrics.py" "${bs_dir}/server.metrics" \
        > "${bs_dir}/metrics_summary.txt" 2>&1

    # Stop server
    kill ${server_pid} 2>/dev/null || true
    kill_port ${port}
    echo "[bs=${bs}] DONE"
}

# ── Launch all batch sizes in parallel waves ─────────────────────────

PIDS=()
NUM_BS=${#BATCH_SIZES[@]}
wave=0

for ((i=0; i<NUM_BS; i+=SLOTS)); do
    wave=$((wave + 1))
    echo ""
    echo "── Wave ${wave}: batch sizes ${BATCH_SIZES[@]:i:SLOTS} ──"

    WAVE_PIDS=()
    for ((j=0; j<SLOTS && (i+j)<NUM_BS; j++)); do
        bs=${BATCH_SIZES[$((i+j))]}
        gpu_group=${GPU_GROUPS[$j]}
        port=$((BASE_PORT + j))

        run_one_bs ${bs} "${gpu_group}" ${port} &
        WAVE_PIDS+=($!)
    done

    # Wait for this wave to complete before starting next
    echo "  Waiting for wave ${wave} (${#WAVE_PIDS[@]} jobs)..."
    for pid in "${WAVE_PIDS[@]}"; do
        wait ${pid} || echo "  WARNING: job ${pid} failed"
    done
    echo "  Wave ${wave} complete."
done

# ── Summary table ────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  SUMMARY: TPOT vs max-num-seqs"
echo "  Model:  ${MODEL_PATH}"
echo "  Config: ${CONFIG_LABEL}"
echo "============================================================"

python3 - "${RUN_DIR}" "${BATCH_SIZES[*]}" <<'PYEOF'
import sys, os, json, glob

run_dir = sys.argv[1]
batch_sizes = sys.argv[2].split()

# Header
print(f"\n{'max_num_seqs':>14s} | {'TPOT p50':>10s} | {'TPOT p90':>10s} | {'TPOT p99':>10s} | {'TTFT p50':>10s} | {'TTFT p99':>10s} | {'Out tok/s':>10s}")
print("-" * 90)

results = []
for bs in batch_sizes:
    bs_dir = os.path.join(run_dir, f"bs_{bs}")
    json_files = glob.glob(os.path.join(bs_dir, "bench_bs*.json"))
    if not json_files:
        print(f"{bs:>14s} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>10s}")
        continue

    with open(json_files[0]) as f:
        data = json.load(f)

    tpot_p50 = data.get("p50_tpot_ms", data.get("median_tpot_ms", 0))
    tpot_p90 = data.get("p90_tpot_ms", 0)
    tpot_p99 = data.get("p99_tpot_ms", 0)
    ttft_p50 = data.get("p50_ttft_ms", data.get("median_ttft_ms", 0))
    ttft_p99 = data.get("p99_ttft_ms", 0)
    out_tps  = data.get("output_throughput", 0)

    print(f"{bs:>14s} | {tpot_p50:>10.2f} | {tpot_p90:>10.2f} | {tpot_p99:>10.2f} | {ttft_p50:>10.2f} | {ttft_p99:>10.2f} | {out_tps:>10.2f}")

    results.append({
        "max_num_seqs": int(bs),
        "tpot_p50_ms": tpot_p50,
        "tpot_p90_ms": tpot_p90,
        "tpot_p99_ms": tpot_p99,
        "ttft_p50_ms": ttft_p50,
        "ttft_p99_ms": ttft_p99,
        "output_throughput": out_tps,
    })

# Save summary JSON
summary_path = os.path.join(run_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump({"model": os.path.basename(run_dir), "results": results}, f, indent=2)
print(f"\nSummary JSON: {summary_path}")
PYEOF

echo ""
echo "All results in: ${RUN_DIR}"
echo "============================================================"
