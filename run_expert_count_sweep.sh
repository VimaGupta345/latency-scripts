#!/bin/bash
#
# Sweep batch sizes and log unique experts activated per layer per batch.
# Uses --enforce-eager so .item() logging works (no CUDA graphs).
#
# Usage:
#   BATCH_SIZES="1 2 4 8 16 32 64" NUM_PROMPTS=256 BASE_PORT=8050 \
#   ./run_expert_count_sweep.sh [model_path] [model_name] [config_file] [tp_size] [gpus]
#
# Example (Qwen3-30B):
#   BATCH_SIZES="1 2 4 8 16 32 64" NUM_PROMPTS=256 BASE_PORT=8050 \
#   ./run_expert_count_sweep.sh \
#       Qwen/Qwen3-30B-A3B-Instruct-2507 qwen3_30b \
#       ../prowl/configs/qwen3/qwen_do-nothing.json 1 "0,1,2,3,4,5,7"

set -euo pipefail

export TMP_HOME="${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}"
source "${TMP_HOME}/prowl/.venv/bin/activate"

MODEL_PATH="${1:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MODEL_NAME="${2:-qwen3_30b}"
CONFIG_FILE="${3:-${TMP_HOME}/prowl/configs/qwen3/qwen_do-nothing.json}"
TP_SIZE="${4:-1}"
GPUS="${5:-0,1,2,3,4,5,7}"

BATCH_SIZES=(${BATCH_SIZES:-1 2 4 8 16 32 64})
NUM_PROMPTS="${NUM_PROMPTS:-256}"
REQUEST_RATE="${REQUEST_RATE:-10000.0}"

DATASET="${DATASET:-${TMP_HOME}/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
BASE_PORT="${BASE_PORT:-8050}"

CONFIG_LABEL=$(basename "${CONFIG_FILE}" .json)
RESULTS_DIR="${TMP_HOME}/results/${MODEL_NAME}_expert_count"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="${RESULTS_DIR}/${CONFIG_LABEL}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

# Track the visualizing_experts dir (relative to where server runs, i.e., this script's CWD)
VIS_DIR="./visualizing_experts"

# ── Parse GPUs into groups of TP_SIZE ────────────────────────────────
IFS=',' read -ra ALL_GPUS <<< "${GPUS}"
NUM_GPUS=${#ALL_GPUS[@]}
SLOTS=$((NUM_GPUS / TP_SIZE))

if [ ${SLOTS} -lt 1 ]; then
    echo "ERROR: Need at least ${TP_SIZE} GPUs for TP=${TP_SIZE}, got ${NUM_GPUS}"
    exit 1
fi

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
echo "  Expert Count Sweep (--enforce-eager)"
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

# ── Single batch-size run ────────────────────────────────────────────

run_one_bs() {
    local bs=$1
    local gpu_group=$2
    local port=$3

    local bs_dir="${RUN_DIR}/bs_${bs}"
    mkdir -p "${bs_dir}"

    echo "[bs=${bs}] Starting on GPU ${gpu_group}, port ${port}"
    kill_port ${port}

    # Record which plots_* dirs exist BEFORE server start
    local pre_dirs=$(ls -d ${VIS_DIR}/plots_* 2>/dev/null | sort)

    # Start vLLM server with --enforce-eager (needed for .item() logging)
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
        --enforce-eager \
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

    # Send requests via vllm bench serve
    echo "[bs=${bs}] Sending ${NUM_PROMPTS} prompts..."
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
        --result-dir "${bs_dir}" \
        --result-filename "bench_bs${bs}.json" \
        > "${bs_dir}/bench.log" 2>&1

    # Stop server (triggers atexit CSV dump)
    kill ${server_pid} 2>/dev/null || true
    sleep 5
    kill_port ${port}

    # Find the NEW plots_* dir created by this run
    local post_dirs=$(ls -d ${VIS_DIR}/plots_* 2>/dev/null | sort)
    local new_dir=$(comm -13 <(echo "${pre_dirs}") <(echo "${post_dirs}") | tail -1)

    if [ -n "${new_dir}" ] && [ -f "${new_dir}/expert_reduction_stats.csv" ]; then
        cp "${new_dir}/expert_reduction_stats.csv" "${bs_dir}/expert_counts.csv"
        echo "[bs=${bs}] Saved expert counts CSV ($(wc -l < "${bs_dir}/expert_counts.csv") rows)"
    else
        echo "[bs=${bs}] WARNING: No expert_reduction_stats.csv found"
    fi

    echo "[bs=${bs}] DONE"
}

# ── Launch in parallel waves ─────────────────────────────────────────

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

    echo "  Waiting for wave ${wave} (${#WAVE_PIDS[@]} jobs)..."
    for pid in "${WAVE_PIDS[@]}"; do
        wait ${pid} || echo "  WARNING: job ${pid} failed"
    done
    echo "  Wave ${wave} complete."
done

# ── Aggregate expert counts ─────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Aggregating expert counts..."
echo "============================================================"

python3 - "${RUN_DIR}" "${BATCH_SIZES[*]}" <<'PYEOF'
import sys, os, csv, json
from collections import defaultdict

run_dir = sys.argv[1]
batch_sizes = sys.argv[2].split()

summary = []

print(f"\n{'batch_size':>12s} | {'num_layers':>10s} | {'avg_unique':>12s} | {'min_unique':>12s} | {'max_unique':>12s} | {'batches':>8s}")
print("-" * 80)

for bs in batch_sizes:
    csv_path = os.path.join(run_dir, f"bs_{bs}", "expert_counts.csv")
    if not os.path.exists(csv_path):
        print(f"{bs:>12s} | {'N/A':>10s} | {'N/A':>12s} | {'N/A':>12s} | {'N/A':>12s} | {'N/A':>8s}")
        continue

    # Parse CSV: columns are batch_idx, layer_idx, original_count, reduced_count, dropped_count, batch_size, reduction_ratio
    per_layer = defaultdict(list)  # layer_idx -> [unique_experts_count, ...]
    batch_ids = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_idx = int(row["layer_idx"])
            unique = int(row["original_count"])
            batch_ids.add(int(row["batch_idx"]))
            per_layer[layer_idx].append(unique)

    num_layers = len(per_layer)
    num_batches = len(batch_ids)

    # Average unique experts per layer, then average across layers
    layer_avgs = {l: sum(v)/len(v) for l, v in per_layer.items()}
    overall_avg = sum(layer_avgs.values()) / len(layer_avgs) if layer_avgs else 0
    overall_min = min(layer_avgs.values()) if layer_avgs else 0
    overall_max = max(layer_avgs.values()) if layer_avgs else 0

    print(f"{bs:>12s} | {num_layers:>10d} | {overall_avg:>12.1f} | {overall_min:>12.1f} | {overall_max:>12.1f} | {num_batches:>8d}")

    summary.append({
        "batch_size": int(bs),
        "num_layers": num_layers,
        "num_batches": num_batches,
        "avg_unique_experts": round(overall_avg, 2),
        "min_layer_avg": round(overall_min, 2),
        "max_layer_avg": round(overall_max, 2),
        "per_layer_avg": {str(l): round(v, 2) for l, v in sorted(layer_avgs.items())},
    })

# Save summary JSON
summary_path = os.path.join(run_dir, "expert_count_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary JSON: {summary_path}")
PYEOF

echo ""
echo "All results in: ${RUN_DIR}"
echo "============================================================"
