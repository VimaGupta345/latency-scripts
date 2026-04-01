#!/bin/bash
#
# Run ShareGPT traffic replay against baseline and prowl configs,
# scraping Prometheus metrics at fixed intervals to build a timeseries
# of TPOT, TTFT, throughput, etc.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 ./run_sharegpt_timeseries.sh [model_path] [model_name] [config_dir]
#
# Example:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 ./run_sharegpt_timeseries.sh openai/gpt-oss-120b gpt_oss_120b prowl/configs/gpt_oss_120b

set -euo pipefail

export TMP_HOME="${TMP_HOME:-/data/vgupta345/prowl_related_data/prowl-open-source}"
source "${TMP_HOME}/prowl/.venv/bin/activate"

# ── Configurable parameters ──────────────────────────────────────────
MODEL_PATH="${1:-openai/gpt-oss-120b}"
MODEL_NAME="${2:-gpt_oss_120b}"
CONFIG_DIR="${3:-${TMP_HOME}/prowl/configs/gpt_oss_120b}"
PORT="${PORT:-8050}"
TP_SIZE="${TP_SIZE:-4}"
REQUEST_RATE="${REQUEST_RATE:-4}"        # requests per second
NUM_PROMPTS="${NUM_PROMPTS:-500}"        # total requests to send
SCRAPE_INTERVAL="${SCRAPE_INTERVAL:-5}"  # seconds between metric scrapes
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"
DATASET="${DATASET:-${TMP_HOME}/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"

RESULTS_DIR="${TMP_HOME}/results/${MODEL_NAME}_sharegpt_timeseries"
mkdir -p "${RESULTS_DIR}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# ── Helper functions ─────────────────────────────────────────────────

kill_port() {
    local port=$1
    local pids=$(lsof -ti:${port} 2>/dev/null)
    if [ -n "${pids}" ]; then
        for pid in ${pids}; do
            pgid=$(ps -o pgid= -p ${pid} 2>/dev/null | tr -d ' ')
            [ -n "${pgid}" ] && kill -9 -${pgid} 2>/dev/null || true
            kill -9 ${pid} 2>/dev/null || true
        done
        sleep 5
    fi
}

wait_for_server() {
    local port=$1
    local max_wait=300
    local elapsed=0
    echo "Waiting for server on port ${port}..."
    while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ ${elapsed} -ge ${max_wait} ]; then
            echo "ERROR: Server did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "Server is ready (took ${elapsed}s)"
}

scrape_metrics_loop() {
    # Scrapes Prometheus metrics every SCRAPE_INTERVAL seconds
    # and appends timestamped snapshots to the output file.
    local port=$1
    local outfile=$2
    local pid_file=$3

    echo "ts_epoch,metric,value" > "${outfile}"
    while true; do
        ts=$(date +%s.%N)
        curl -s "http://localhost:${port}/metrics" 2>/dev/null | \
        grep -E '^vllm:' | \
        grep -v '^#' | \
        while IFS= read -r line; do
            metric_name=$(echo "$line" | awk '{print $1}')
            metric_val=$(echo "$line" | awk '{print $NF}')
            echo "${ts},${metric_name},${metric_val}"
        done >> "${outfile}"
        sleep ${SCRAPE_INTERVAL}
    done &
    echo $! > "${pid_file}"
}

run_one_config() {
    local config_file=$1
    local config_label=$2

    echo ""
    echo "============================================================"
    echo "  Config: ${config_label}"
    echo "  Model:  ${MODEL_PATH}"
    echo "  Rate:   ${REQUEST_RATE} RPS, ${NUM_PROMPTS} prompts"
    echo "============================================================"

    # Kill any leftover server
    kill_port ${PORT}

    # Start vLLM server
    local server_log="${RESULTS_DIR}/${config_label}_server_${TIMESTAMP}.log"
    python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --host localhost \
        --port ${PORT} \
        --max-num-seqs ${MAX_BATCH_SIZE} \
        --tensor-parallel-size ${TP_SIZE} \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.9 \
        --mixtral_config_file ${config_file} \
        --trust-remote-code \
        2>&1 | tee "${server_log}" &
    local server_pid=$!

    wait_for_server ${PORT}

    # Start metrics scraper
    local metrics_ts="${RESULTS_DIR}/${config_label}_metrics_ts_${TIMESTAMP}.csv"
    local scraper_pid_file="/tmp/scraper_pid_${PORT}"
    scrape_metrics_loop ${PORT} "${metrics_ts}" "${scraper_pid_file}"
    echo "Metrics scraper writing to: ${metrics_ts}"

    # Run ShareGPT benchmark
    local bench_output="${RESULTS_DIR}/${config_label}_bench_${TIMESTAMP}.json"
    cd "${TMP_HOME}/prowl/benchmarks"
    python benchmark_serving.py \
        --backend openai-chat \
        --model "${MODEL_PATH}" \
        --base-url "http://localhost:${PORT}" \
        --endpoint "/v1/chat/completions" \
        --dataset-name sharegpt \
        --dataset-path "${DATASET}" \
        --request-rate ${REQUEST_RATE} \
        --num-prompts ${NUM_PROMPTS} \
        --save-result \
        --result-dir "${RESULTS_DIR}" \
        --result-filename "${config_label}_bench_${TIMESTAMP}.json" \
        --percentile-metrics ttft,tpot,itl,e2el \
        2>&1 | tee "${RESULTS_DIR}/${config_label}_bench_${TIMESTAMP}.log"
    cd "${TMP_HOME}/latency-scripts"

    # Scrape final metrics snapshot
    curl -s "http://localhost:${PORT}/metrics" > \
        "${RESULTS_DIR}/${config_label}_final_${TIMESTAMP}.metrics"

    # Stop scraper and server
    [ -f "${scraper_pid_file}" ] && kill $(cat "${scraper_pid_file}") 2>/dev/null || true
    kill ${server_pid} 2>/dev/null || true
    kill_port ${PORT}

    echo "  Done: ${config_label}"
}

# ── Main ─────────────────────────────────────────────────────────────

configs=(
    "${CONFIG_DIR}/gpt_oss_do-nothing.json:baseline"
    "${CONFIG_DIR}/quant_alpha3_beta2_optimized.json:prowl"
)

echo "ShareGPT Timeseries Benchmark"
echo "  Model:     ${MODEL_PATH}"
echo "  Port:      ${PORT}"
echo "  TP:        ${TP_SIZE}"
echo "  Rate:      ${REQUEST_RATE} RPS"
echo "  Prompts:   ${NUM_PROMPTS}"
echo "  Scrape:    every ${SCRAPE_INTERVAL}s"
echo "  Results:   ${RESULTS_DIR}"
echo ""

for entry in "${configs[@]}"; do
    config_file="${entry%%:*}"
    config_label="${entry##*:}"
    run_one_config "${config_file}" "${config_label}"
done

# ── Generate comparison plot ─────────────────────────────────────────
echo ""
echo "Generating timeseries comparison..."

python3 - "${RESULTS_DIR}" "${TIMESTAMP}" <<'PYEOF'
import sys, os, csv, glob
import json
from collections import defaultdict

results_dir = sys.argv[1]
timestamp = sys.argv[2]

# ── Parse timeseries metrics CSVs ──
def parse_metrics_ts(filepath):
    """Parse scraped metrics CSV into {metric: [(relative_time, value), ...]}"""
    series = defaultdict(list)
    t0 = None
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["ts_epoch"])
            if t0 is None:
                t0 = ts
            rel_t = ts - t0
            series[row["metric"]].append((rel_t, float(row["value"])))
    return series

# ── Parse per-request data from benchmark JSON ──
def parse_bench_json(filepath):
    """Extract per-request TPOT and TTFT from benchmark output."""
    with open(filepath) as f:
        data = json.load(f)
    return data

# ── Find files ──
baseline_ts = glob.glob(f"{results_dir}/baseline_metrics_ts_{timestamp}.csv")
prowl_ts = glob.glob(f"{results_dir}/prowl_metrics_ts_{timestamp}.csv")
baseline_bench = glob.glob(f"{results_dir}/baseline_bench_{timestamp}.json")
prowl_bench = glob.glob(f"{results_dir}/prowl_bench_{timestamp}.json")

# ── Print summary ──
print("\n" + "=" * 70)
print("  SHAREGPT TIMESERIES BENCHMARK SUMMARY")
print("=" * 70)

for label, bench_files in [("Baseline", baseline_bench), ("Prowl", prowl_bench)]:
    if not bench_files or not os.path.exists(bench_files[0]):
        print(f"\n  {label}: no results found")
        continue
    data = parse_bench_json(bench_files[0])
    print(f"\n  {label}:")
    print(f"    Completed requests:   {data.get('completed', 'N/A')}")
    print(f"    Duration (s):         {data.get('duration', 0):.1f}")
    print(f"    Request throughput:    {data.get('request_throughput', 0):.2f} req/s")
    print(f"    Output throughput:     {data.get('output_throughput', 0):.2f} tok/s")
    for key in ["mean_ttft_ms", "median_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
                "mean_itl_ms", "median_itl_ms", "mean_e2el_ms", "median_e2el_ms"]:
        if key in data:
            print(f"    {key:25s} {data[key]:.2f}")

# ── Compute speedup ──
if baseline_bench and prowl_bench:
    b = parse_bench_json(baseline_bench[0]) if os.path.exists(baseline_bench[0]) else {}
    p = parse_bench_json(prowl_bench[0]) if os.path.exists(prowl_bench[0]) else {}
    print(f"\n  {'─' * 50}")
    print(f"  PROWL Speedup:")
    for key, label in [("median_tpot_ms", "TPOT (median)"),
                       ("mean_tpot_ms", "TPOT (mean)"),
                       ("median_ttft_ms", "TTFT (median)"),
                       ("median_itl_ms", "ITL (median)"),
                       ("output_throughput", "Output tok/s")]:
        bv = b.get(key, 0)
        pv = p.get(key, 0)
        if bv and pv:
            if "throughput" in key:
                ratio = pv / bv
            else:
                ratio = bv / pv
            print(f"    {label:25s} {ratio:.3f}x  (base={bv:.2f}, prowl={pv:.2f})")

# ── Write timeseries CSVs for plotting ──
for label, ts_files in [("baseline", baseline_ts), ("prowl", prowl_ts)]:
    if not ts_files or not os.path.exists(ts_files[0]):
        continue
    series = parse_metrics_ts(ts_files[0])

    # Extract key metrics as separate CSVs for easy plotting
    key_metrics = [
        "vllm:time_per_output_token_seconds_sum",
        "vllm:time_per_output_token_seconds_count",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:gpu_cache_usage_perc",
    ]
    summary_file = f"{results_dir}/{label}_timeseries_summary_{timestamp}.csv"
    with open(summary_file, "w") as f:
        f.write("time_s," + ",".join(k.replace("vllm:", "") for k in key_metrics) + "\n")
        # Align by time buckets
        all_times = sorted(set(t for k in key_metrics for t, v in series.get(k, [])))
        latest = {k: 0.0 for k in key_metrics}
        for t in all_times:
            for k in key_metrics:
                for ts, val in series.get(k, []):
                    if ts == t:
                        latest[k] = val
            f.write(f"{t:.1f}," + ",".join(f"{latest[k]}" for k in key_metrics) + "\n")
    print(f"\n  Timeseries CSV: {summary_file}")

print(f"\n  All results in: {results_dir}")
print("=" * 70)
PYEOF

echo ""
echo "Done! Results in: ${RESULTS_DIR}"
