#!/usr/bin/env python3
"""
Gather experiment settings + Prowl config + results (accuracy, latency) into a single CSV.

Covers three result types:
  1. lm-eval results (colocated + disagg) — accuracy + TPOT from .metrics histograms
  2. Batch-size sweep results — direct TPOT/ITL from bench_bs*.json
  3. AIPerf trace results — direct ITL/TTFT from CSV files

Latency columns are split:
  - *_direct: values read directly from bench_bs*.json or AIPerf CSV
  - *_hist:   values estimated from Prometheus histogram bucket interpolation

Usage:
    # Default: scan ./results and write ./results/all_results.csv
    python latency-scripts/gather_all_results.py

    # Custom output path
    python latency-scripts/gather_all_results.py --output /tmp/my_results.csv

    # Point at a different results directory
    python latency-scripts/gather_all_results.py /path/to/results -o out.csv

Example output columns:
    model, benchmark, nshot, tp, num_gpus, batch_size, mode,
    cuda_graph_enabled, cudagraph_mode, config, is_baseline,
    accuracy, accuracy_metric, accuracy_delta,
    p50_tpot_ms_direct, p90_tpot_ms_direct, p99_tpot_ms_direct, mean_tpot_ms_direct,
    p50_tpot_ms_hist_fine, ..., p50_tpot_ms_hist_coarse, ...,
    p50_itl_ms_direct, p90_itl_ms_direct, p99_itl_ms_direct, mean_itl_ms_direct,
    p50_tpot_speedup_direct, ..., p50_tpot_speedup_hist, ...,
    source
"""

import argparse
import csv
import json
import glob
import os
import re


# ── Known TP mappings (fallback; overridden by server_logs when available) ─
KNOWN_TP = {
    "DeepSeek-Coder-V2-Instruct": 4,
    "DeepSeek-Coder-V2-Instruct-FP8": 4,
    "DeepSeek-R1": 4,
    "DeepSeek-V3": 4,
    "Mixtral-8x7B-Instruct-v0.1": 2,
    "Mixtral-8x22B-Instruct-v0.1": 4,
    "Qwen2-57B-A14B-Instruct": 2,
    "Qwen2-57B-A14B-Instruct-GPTQ-Int4": 2,
    "Qwen3-30B-A3B-Instruct-2507": 1,
    "Qwen3-30B-A3B-Thinking": 1,
    "Qwen3-30B-A3B-Thinking-2507": 1,
    "Qwen3-235B-A22B-Thinking-2507": 4,
    "gpt-oss-120b": 4,
    "Llama-4-Scout-17B-16E-Instruct": 4,
    "Qwen3-32B": 1,
    "Qwen3-4B": 1,
}

# ── N-shot defaults per benchmark (from lm_eval_online_serve.py) ────────
NSHOT_DEFAULTS = {
    "humaneval": 0, "mbpp": 3, "gsm8k": 5, "minerva_math_algebra": 4,
    "truthfulqa": 0, "truthfulqa_mc2": 0, "triviaqa": 0, "coqa": 0,
    "hotpotqa": 0, "cnn_dailymail": 0, "xsum": 0, "squad_completion": 0,
    "squadv2": 0, "longbench_narrativeqa": 0, "aime24": 0, "aime25": 0,
    "aime_2024": 0, "chartqa": 0, "mt_bench": 0,
}

# ── Accuracy metric markers per benchmark ───────────────────────────────
ACCURACY_MARKERS = {
    "humaneval": "pass@1,", "mbpp": "pass_at_1,",
    "minerva_math_algebra": "math_verify,", "gsm8k": "exact_match,flexible",
    "truthfulqa": "rougeL_acc,", "truthfulqa_mc2": "acc,",
    "squad_completion": "contains,", "longbench_narrativeqa": "qa_f1_score,",
    "coqa": "em,", "cnn_dailymail": "rouge,", "xsum": "rouge,",
    "hotpotqa": "qa_f1_score,", "triviaqa": "qa_f1_score,",
    "squadv2": "exact,", "aime24": "exact_match,", "aime25": "exact_match,",
    "aime_2024": "exact_match,", "chartqa": "exact_match,",
}

# ── Baseline config name patterns ──────────────────────────────────────
BASELINE_PATTERNS = [
    "config_do_nothing", "do_nothing", "do-nothing",
    "qwen_do-nothing", "qwen_do_nothing",
    "mixtral_do_nothing", "deepseek_do_nothing",
    "no_drop",
]


def is_baseline(config_name):
    normalized = config_name.lower().replace("-", "_")
    for pat in BASELINE_PATTERNS:
        if pat.replace("-", "_") in normalized:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────
#  Prometheus histogram parsing
# ─────────────────────────────────────────────────────────────────────────

def parse_histogram(filepath, metric_prefix):
    buckets, count, total = [], None, None
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{metric_prefix}_bucket{{"):
                    le_match = re.search(r'le="([^"]+)"', line)
                    val = float(line.split()[-1])
                    le = le_match.group(1)
                    le = float("inf") if le == "+Inf" else float(le)
                    buckets.append((le, val))
                elif line.startswith(f"{metric_prefix}_count{{"):
                    count = float(line.split()[-1])
                elif line.startswith(f"{metric_prefix}_sum{{"):
                    total = float(line.split()[-1])
    except (IOError, OSError):
        pass
    return buckets, count, total


def percentile_from_buckets(buckets, count, p):
    if not buckets or not count or count == 0:
        return None
    target = count * p
    prev_le, prev_count = 0.0, 0.0
    for le, cum_count in buckets:
        if le == float("inf"):
            return prev_le
        if cum_count >= target:
            bucket_count = cum_count - prev_count
            if bucket_count == 0:
                return le
            fraction = (target - prev_count) / bucket_count
            return prev_le + fraction * (le - prev_le)
        prev_le, prev_count = le, cum_count
    return prev_le


def classify_histogram_granularity(buckets):
    """Classify histogram as 'fine' (1ms steps, ~109 buckets) or 'coarse' (19 buckets).
    Returns ('fine', n_buckets) or ('coarse', n_buckets)."""
    n = len([b for b in buckets if b[0] != float("inf")])
    return ("fine" if n > 50 else "coarse", n)


def extract_tpot_from_metrics(metrics_path):
    """Returns dict with p50/p90/p99/mean_tpot_ms_hist (estimated from histogram),
    split into _hist_fine and _hist_coarse columns based on bucket granularity."""
    prefix = "vllm:time_per_output_token_seconds"
    buckets, count, total = parse_histogram(metrics_path, prefix)
    if not buckets or not count or count == 0:
        return {}

    granularity, n_buckets = classify_histogram_granularity(buckets)
    suffix = f"_hist_{granularity}"

    result = {}
    result["hist_granularity"] = granularity
    result["hist_n_buckets"] = n_buckets

    mean = total / count
    result[f"mean_tpot_ms{suffix}"] = mean * 1000
    for p, label in [(0.50, "p50"), (0.90, "p90"), (0.99, "p99")]:
        val = percentile_from_buckets(buckets, count, p)
        if val is not None:
            result[f"{label}_tpot_ms{suffix}"] = val * 1000
    return result


# ─────────────────────────────────────────────────────────────────────────
#  Accuracy extraction from results_*.json
# ─────────────────────────────────────────────────────────────────────────

def extract_accuracy(results_json_path, benchmark_name):
    try:
        with open(results_json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None, None, None

    if "results" not in data:
        return None, None, None

    # nshot
    nshot = None
    if "n-shot" in data:
        for bmk, val in data["n-shot"].items():
            nshot = val
            break
    if nshot is None and "configs" in data:
        for bmk, cfg in data["configs"].items():
            if "num_fewshot" in cfg:
                nshot = cfg["num_fewshot"]
                break

    # accuracy
    accuracy, metric_name = None, None
    for bmk_name, bmk_results in data["results"].items():
        marker = ACCURACY_MARKERS.get(bmk_name) or ACCURACY_MARKERS.get(benchmark_name)
        if marker:
            for key, val in bmk_results.items():
                if key.startswith(marker) and isinstance(val, (int, float)):
                    accuracy, metric_name = val, key
                    break
        if accuracy is None:
            for key, val in bmk_results.items():
                if isinstance(val, (int, float)) and "stderr" not in key and key != "alias":
                    accuracy, metric_name = val, key
                    break
    return accuracy, metric_name, nshot


# ─────────────────────────────────────────────────────────────────────────
#  Server log parsing: extract TP, batch_size, enforce_eager, cudagraph
# ─────────────────────────────────────────────────────────────────────────

def parse_server_log(log_path):
    """Extract settings from a vLLM server log file."""
    info = {}
    try:
        with open(log_path) as f:
            content = f.read(50000)  # first 50KB is enough
    except (IOError, OSError):
        return info

    # non-default args line: has tp, max_num_seqs, port, model, config
    m = re.search(r"non-default args:\s*(\{.*?\})", content)
    if m:
        try:
            args = eval(m.group(1))  # safe: only contains simple types
            info["tp"] = args.get("tensor_parallel_size")
            info["max_num_seqs"] = args.get("max_num_seqs")
        except Exception:
            pass

    # Fallback: if TP not in non-default args (i.e. TP=1 default), try direct regex
    if info.get("tp") is None:
        tp_match = re.search(r"tensor_parallel_size=(\d+)", content)
        info["tp"] = int(tp_match.group(1)) if tp_match else 1  # vLLM default is 1

    # enforce_eager
    m = re.search(r"enforce_eager=(\w+)", content)
    if m:
        info["enforce_eager"] = m.group(1) == "True"

    # cudagraph
    m = re.search(r'"use_cudagraph":(\w+)', content)
    if m:
        info["use_cudagraph"] = m.group(1) == "true"
    m = re.search(r'"cudagraph_mode":(\d+)', content)
    if m:
        info["cudagraph_mode"] = int(m.group(1))

    return info


def build_server_log_index(results_dir):
    """Build index: (model_dir, benchmark, config) -> server log info."""
    index = {}

    # 1. Parse results/server_logs/{model}/*.log (colocated lm-eval runs)
    server_logs_dir = os.path.join(results_dir, "server_logs")
    if os.path.isdir(server_logs_dir):
        for model_dir in os.listdir(server_logs_dir):
            model_path = os.path.join(server_logs_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
            for logfile in os.listdir(model_path):
                if not logfile.endswith(".log"):
                    continue
                m = re.match(r"(\w+?)_ndef_.*_conf_(.+?)\.json_port\d+\.log$", logfile)
                if not m:
                    m = re.match(r"(\w+?)_ndef_.*_port\d+_(.+?)\.json_port\d+\.log$", logfile)
                if m:
                    benchmark = m.group(1)
                    config = re.sub(r"\.json$", "", m.group(2))
                    info = parse_server_log(os.path.join(model_path, logfile))
                    index[(model_dir, benchmark, config)] = info
                else:
                    info = parse_server_log(os.path.join(model_path, logfile))
                    if info.get("tp"):
                        index[(model_dir, "_fallback", "_fallback")] = info

    # 2. Parse disagg decode logs for TP and cuda_graph info
    # Filenames: decode_{model_short}_{config}_{benchmark}_{timestamp}.log
    # Index by timestamp so we can match to specific result files.
    for subdir in ("disagg_sweep", "disagg"):
        decode_log_dir = os.path.join(results_dir, "logs", subdir)
        if not os.path.isdir(decode_log_dir):
            continue
        for logfile in sorted(os.listdir(decode_log_dir)):
            if not logfile.startswith("decode_") or not logfile.endswith(".log"):
                continue
            # Extract timestamp from filename
            ts_match = re.search(r"(\d{8}-\d{6})\.log$", logfile)
            if ts_match:
                ts = ts_match.group(1)
                info = parse_server_log(os.path.join(decode_log_dir, logfile))
                if info:
                    index[("_disagg_ts", ts)] = info

    return index


# Map model dir names to short names used in disagg decode log filenames
_MODEL_DIR_TO_SHORT = {
    "DeepSeek-Coder-V2-Instruct": "deepseek_v2",
    "Mixtral-8x7B-Instruct-v0.1": "mixtral",
    "Qwen2-57B-A14B-Instruct": "qwen",
    "Qwen3-30B-A3B-Instruct-2507": "qwen3",
    "Qwen3-30B-A3B-Thinking-2507": "qwen3",
}


def lookup_server_info(index, model_dir, benchmark, config,
                       is_disagg=False, timestamp=None):
    """Look up server info, with fallback to model-level defaults."""
    # For disagg: try timestamp-based decode log match first (most accurate)
    if is_disagg and timestamp:
        info = index.get(("_disagg_ts", timestamp))
        if info:
            return info
    # Try exact match from server_logs/
    info = index.get((model_dir, benchmark, config))
    if info:
        return info
    # Try model-level fallback
    info = index.get((model_dir, "_fallback", "_fallback"))
    if info:
        return info
    return {}


# ─────────────────────────────────────────────────────────────────────────
#  Orchestration log parsing (for disagg GPU info)
# ─────────────────────────────────────────────────────────────────────────

def parse_orchestration_logs(logs_dir):
    """Parse orchestration logs to update KNOWN_TP and get disagg GPU info."""
    disagg_gpu_info = {}

    if not os.path.isdir(logs_dir):
        return disagg_gpu_info

    for logfile in glob.glob(os.path.join(logs_dir, "*.log")):
        try:
            with open(logfile) as f:
                content = f.read(10000)
        except IOError:
            continue
        for m in re.finditer(r"MODEL:\s*\w+\s*\(([^)]+)\),\s*TP=(\d+)", content):
            model_name = m.group(1).split("/")[-1]
            KNOWN_TP[model_name] = int(m.group(2))

    # Disagg sweep logs
    disagg_sweep_dir = os.path.join(logs_dir, "disagg_sweep")
    if os.path.isdir(disagg_sweep_dir):
        for logfile in glob.glob(os.path.join(disagg_sweep_dir, "sweep_*.log")):
            try:
                with open(logfile) as f:
                    content = f.read(5000)
            except IOError:
                continue
            pm = re.search(r"Prefill GPUs?:\s*([0-9,]+)", content)
            dm = re.search(r"Decode GPUs?:\s*([0-9,]+)", content)
            if pm:
                disagg_gpu_info["prefill_gpus"] = pm.group(1)
            if dm:
                disagg_gpu_info["decode_gpus"] = dm.group(1)
            for m in re.finditer(r"MODEL:\s*\w+\s*\(([^)]+)\),\s*TP=(\d+)", content):
                model_name = m.group(1).split("/")[-1]
                KNOWN_TP[model_name] = int(m.group(2))

    return disagg_gpu_info


# ─────────────────────────────────────────────────────────────────────────
#  Find corresponding .metrics file
# ─────────────────────────────────────────────────────────────────────────

def _metrics_conf_key(name):
    """Extract the _conf_XXX portion from a filename for matching."""
    m = re.search(r"_conf_(.+?)(?:_\d{8}-\d{6})?(?:_(?:decode|prefill))?\.(?:metrics|jsonl)$", name)
    return m.group(1) if m else None


def find_metrics_file(result_dir, base_name, is_disagg):
    candidates = []
    try:
        entries = os.listdir(result_dir)
    except OSError:
        return None

    base_no_ts = re.sub(r"_\d{8}-\d{6}$", "", base_name)
    conf_key = _metrics_conf_key(base_name + ".jsonl")  # reuse extraction

    if is_disagg:
        # Exact prefix match first
        for f in entries:
            if f.startswith(base_name) and f.endswith("_decode.metrics"):
                return os.path.join(result_dir, f)
        for f in entries:
            if f.endswith("_decode.metrics") and base_no_ts in f:
                return os.path.join(result_dir, f)
        # Fall back to config-based matching for disagg
        if conf_key:
            for f in entries:
                if f.endswith("_decode.metrics") and f"_conf_{conf_key}" in f:
                    candidates.append(os.path.join(result_dir, f))
    else:
        for f in entries:
            if f.endswith(".metrics") and not f.endswith("_prefill.metrics") and \
               not f.endswith("_decode.metrics"):
                if f.startswith(base_no_ts):
                    candidates.append(os.path.join(result_dir, f))

        # Fallback: match on _conf_ key when limit (nXXX) differs between
        # .jsonl dir name and .metrics filename (e.g. n0.0 vs n250)
        if not candidates and conf_key:
            for f in entries:
                if f.endswith(".metrics") and not f.endswith("_prefill.metrics") and \
                   not f.endswith("_decode.metrics"):
                    if f"_conf_{conf_key}" in f:
                        candidates.append(os.path.join(result_dir, f))

    if candidates:
        candidates.sort()
        return candidates[-1]
    return None


# ─────────────────────────────────────────────────────────────────────────
#  Gather lm-eval results (colocated + disagg)
# ─────────────────────────────────────────────────────────────────────────

def gather_lm_eval_results(results_dir, server_log_index, disagg_gpu_info):
    rows = []
    skip_dirs = {"logs", "experiment_logs", "server_logs", ".claude"}

    for model_dir_name in sorted(os.listdir(results_dir)):
        model_path = os.path.join(results_dir, model_dir_name)
        if not os.path.isdir(model_path) or model_dir_name in skip_dirs:
            continue
        if any(model_dir_name.endswith(s) for s in
               ("_bs_sweep", "_prowl_bs_sweep", "_expert_count",
                "_sharegpt_timeseries", "_multimodal")) or \
           model_dir_name.startswith("qwen_aiperf_"):
            continue

        for benchmark_dir_name in sorted(os.listdir(model_path)):
            benchmark_path = os.path.join(model_path, benchmark_dir_name)
            if not os.path.isdir(benchmark_path):
                continue

            for entry in os.listdir(benchmark_path):
                entry_path = os.path.join(benchmark_path, entry)
                if not entry.endswith(".jsonl") or not os.path.isdir(entry_path):
                    continue

                is_disagg = entry.startswith("disagg_nixl_") or entry.startswith("disagg_")
                mode = "disagg" if is_disagg else "colocated"

                # Config name and timestamp
                config_match = re.search(r"_conf_(.+)\.jsonl$", entry)
                config_name = config_match.group(1) if config_match else "unknown"
                # Extract timestamp before stripping it from config
                ts_match = re.search(r"_(\d{8}-\d{6})$", config_name)
                entry_timestamp = ts_match.group(1) if ts_match else None
                config_name = re.sub(r"_\d{8}-\d{6}$", "", config_name)

                # Find results_*.json
                results_json = None
                for root, dirs, files in os.walk(entry_path):
                    for f in files:
                        if f.startswith("results_") and f.endswith(".json"):
                            results_json = os.path.join(root, f)
                            break
                    if results_json:
                        break
                if not results_json:
                    continue

                accuracy, metric_name, nshot = extract_accuracy(results_json, benchmark_dir_name)
                if nshot is None:
                    nshot = NSHOT_DEFAULTS.get(benchmark_dir_name, "")

                # TPOT from .metrics (histogram-estimated)
                base_name = entry.replace(".jsonl", "")
                metrics_file = find_metrics_file(benchmark_path, base_name, is_disagg)
                tpot_hist = extract_tpot_from_metrics(metrics_file) if metrics_file else {}

                # Server log info
                srv = lookup_server_info(server_log_index, model_dir_name,
                                         benchmark_dir_name, config_name,
                                         is_disagg=is_disagg,
                                         timestamp=entry_timestamp)
                tp = srv.get("tp") or KNOWN_TP.get(model_dir_name, "")
                batch_size = srv.get("max_num_seqs", 16)
                enforce_eager = srv.get("enforce_eager")
                use_cudagraph = srv.get("use_cudagraph")
                cudagraph_mode = srv.get("cudagraph_mode")

                # Derive cuda_graph_enabled
                if enforce_eager is True:
                    cuda_graph_enabled = False
                elif use_cudagraph is not None:
                    cuda_graph_enabled = use_cudagraph
                elif enforce_eager is False:
                    cuda_graph_enabled = True
                else:
                    cuda_graph_enabled = ""  # unknown

                # num_gpus
                if is_disagg:
                    num_gpus = tp * 2 if tp else ""
                else:
                    num_gpus = tp if tp else ""

                rows.append({
                    "model": model_dir_name,
                    "benchmark": benchmark_dir_name,
                    "nshot": nshot if nshot is not None else "",
                    "tp": tp,
                    "num_gpus": num_gpus,
                    "batch_size": batch_size,
                    "mode": mode,
                    "cuda_graph_enabled": cuda_graph_enabled,
                    "cudagraph_mode": cudagraph_mode if cudagraph_mode is not None else "",
                    "config": config_name,
                    "accuracy": accuracy,
                    "accuracy_metric": metric_name or "",
                    # Direct values: not available for lm-eval
                    "p50_tpot_ms_direct": "",
                    "p90_tpot_ms_direct": "",
                    "p99_tpot_ms_direct": "",
                    "mean_tpot_ms_direct": "",
                    "p50_itl_ms_direct": "",
                    "p90_itl_ms_direct": "",
                    "p99_itl_ms_direct": "",
                    "mean_itl_ms_direct": "",
                    # Histogram-estimated values — fine (1ms buckets) or coarse (19 buckets)
                    "hist_granularity": tpot_hist.get("hist_granularity", ""),
                    "hist_n_buckets": tpot_hist.get("hist_n_buckets", ""),
                    "p50_tpot_ms_hist_fine": tpot_hist.get("p50_tpot_ms_hist_fine", ""),
                    "p90_tpot_ms_hist_fine": tpot_hist.get("p90_tpot_ms_hist_fine", ""),
                    "p99_tpot_ms_hist_fine": tpot_hist.get("p99_tpot_ms_hist_fine", ""),
                    "mean_tpot_ms_hist_fine": tpot_hist.get("mean_tpot_ms_hist_fine", ""),
                    "p50_tpot_ms_hist_coarse": tpot_hist.get("p50_tpot_ms_hist_coarse", ""),
                    "p90_tpot_ms_hist_coarse": tpot_hist.get("p90_tpot_ms_hist_coarse", ""),
                    "p99_tpot_ms_hist_coarse": tpot_hist.get("p99_tpot_ms_hist_coarse", ""),
                    "mean_tpot_ms_hist_coarse": tpot_hist.get("mean_tpot_ms_hist_coarse", ""),
                    "source": "lm-eval",
                })

    return rows


# ─────────────────────────────────────────────────────────────────────────
#  Gather batch-size sweep results
# ─────────────────────────────────────────────────────────────────────────

def infer_model_from_sweep_dir(dirname):
    mapping = {
        "qwen2_57b": "Qwen2-57B-A14B-Instruct",
        "qwen3_30b": "Qwen3-30B-A3B-Instruct-2507",
        "qwen3_32b": "Qwen3-32B",
        "qwen3_4b": "Qwen3-4B",
        "qwen3_235b": "Qwen3-235B-A22B-Thinking-2507",
    }
    for prefix, model in mapping.items():
        if dirname.startswith(prefix):
            return model
    return dirname


def gather_bs_sweep_results(results_dir):
    rows = []

    for sweep_dir_name in sorted(os.listdir(results_dir)):
        sweep_path = os.path.join(results_dir, sweep_dir_name)
        if not os.path.isdir(sweep_path):
            continue
        if not (sweep_dir_name.endswith("_bs_sweep") or
                sweep_dir_name.endswith("_prowl_bs_sweep") or
                sweep_dir_name.endswith("_expert_count")):
            continue

        model_name = infer_model_from_sweep_dir(sweep_dir_name)

        for config_ts_dir in sorted(os.listdir(sweep_path)):
            config_ts_path = os.path.join(sweep_path, config_ts_dir)
            if not os.path.isdir(config_ts_path):
                continue

            config_name = re.sub(r"_\d{8}-\d{6}$", "", config_ts_dir)

            for bs_dir in sorted(os.listdir(config_ts_path)):
                bs_path = os.path.join(config_ts_path, bs_dir)
                if not os.path.isdir(bs_path) or not bs_dir.startswith("bs_"):
                    continue

                batch_size = int(bs_dir.replace("bs_", ""))

                # bench_bs*.json for direct values
                bench_files = glob.glob(os.path.join(bs_path, "bench_bs*.json"))
                bench_data = {}
                if bench_files:
                    try:
                        with open(bench_files[0]) as f:
                            bench_data = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        pass

                # server.log for TP, cudagraph
                server_log = os.path.join(bs_path, "server.log")
                srv = parse_server_log(server_log) if os.path.isfile(server_log) else {}

                tp = srv.get("tp") or KNOWN_TP.get(model_name, "")
                enforce_eager = srv.get("enforce_eager")
                use_cudagraph = srv.get("use_cudagraph")
                cudagraph_mode = srv.get("cudagraph_mode")

                if enforce_eager is True:
                    cuda_graph_enabled = False
                elif use_cudagraph is not None:
                    cuda_graph_enabled = use_cudagraph
                elif enforce_eager is False:
                    cuda_graph_enabled = True
                else:
                    cuda_graph_enabled = ""

                # TPOT from server.metrics (histogram)
                server_metrics = os.path.join(bs_path, "server.metrics")
                tpot_hist = extract_tpot_from_metrics(server_metrics) if os.path.isfile(server_metrics) else {}

                rows.append({
                    "model": model_name or sweep_dir_name,
                    "benchmark": "sharegpt",
                    "nshot": "",
                    "tp": tp,
                    "num_gpus": tp if tp else "",
                    "batch_size": batch_size,
                    "mode": "colocated",
                    "cuda_graph_enabled": cuda_graph_enabled,
                    "cudagraph_mode": cudagraph_mode if cudagraph_mode is not None else "",
                    "config": config_name,
                    "accuracy": "",
                    "accuracy_metric": "",
                    # Direct values from bench_bs*.json
                    "p50_tpot_ms_direct": bench_data.get("p50_tpot_ms", ""),
                    "p90_tpot_ms_direct": bench_data.get("p90_tpot_ms", ""),
                    "p99_tpot_ms_direct": bench_data.get("p99_tpot_ms", ""),
                    "mean_tpot_ms_direct": bench_data.get("mean_tpot_ms", ""),
                    "p50_itl_ms_direct": bench_data.get("p50_itl_ms", ""),
                    "p90_itl_ms_direct": bench_data.get("p90_itl_ms", ""),
                    "p99_itl_ms_direct": bench_data.get("p99_itl_ms", ""),
                    "mean_itl_ms_direct": bench_data.get("mean_itl_ms", ""),
                    # Histogram-estimated from server.metrics
                    "hist_granularity": tpot_hist.get("hist_granularity", ""),
                    "hist_n_buckets": tpot_hist.get("hist_n_buckets", ""),
                    "p50_tpot_ms_hist_fine": tpot_hist.get("p50_tpot_ms_hist_fine", ""),
                    "p90_tpot_ms_hist_fine": tpot_hist.get("p90_tpot_ms_hist_fine", ""),
                    "p99_tpot_ms_hist_fine": tpot_hist.get("p99_tpot_ms_hist_fine", ""),
                    "mean_tpot_ms_hist_fine": tpot_hist.get("mean_tpot_ms_hist_fine", ""),
                    "p50_tpot_ms_hist_coarse": tpot_hist.get("p50_tpot_ms_hist_coarse", ""),
                    "p90_tpot_ms_hist_coarse": tpot_hist.get("p90_tpot_ms_hist_coarse", ""),
                    "p99_tpot_ms_hist_coarse": tpot_hist.get("p99_tpot_ms_hist_coarse", ""),
                    "mean_tpot_ms_hist_coarse": tpot_hist.get("mean_tpot_ms_hist_coarse", ""),
                    "source": "bs_sweep",
                })

    return rows


# ─────────────────────────────────────────────────────────────────────────
#  Gather AIPerf trace results
# ─────────────────────────────────────────────────────────────────────────

def parse_aiperf_csv(csv_path):
    """Parse an AIPerf CSV and extract ITL/TTFT percentiles."""
    result = {}
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row.get("Metric", "").strip()
                if metric == "Inter Token Latency (ms)":
                    for col in ("avg", "p50", "p90", "p99"):
                        try:
                            result[f"{'mean' if col == 'avg' else col}_itl_ms_direct"] = float(row[col])
                        except (ValueError, KeyError):
                            pass
                elif metric == "Time to First Token (ms)":
                    for col in ("avg", "p50", "p90", "p99"):
                        try:
                            result[f"{'mean' if col == 'avg' else col}_ttft_ms_direct"] = float(row[col])
                        except (ValueError, KeyError):
                            pass
    except (IOError, OSError):
        pass
    return result


def gather_aiperf_results(results_dir):
    rows = []

    for dir_name in sorted(os.listdir(results_dir)):
        dir_path = os.path.join(results_dir, dir_name)
        if not os.path.isdir(dir_path) or not dir_name.startswith("qwen_aiperf_"):
            continue

        # Determine mode from dir name
        is_disagg = "disagg" in dir_name

        # Infer model (these are all Qwen2-57B runs)
        model_name = "Qwen2-57B-A14B-Instruct"

        for run_dir_name in sorted(os.listdir(dir_path)):
            run_path = os.path.join(dir_path, run_dir_name)
            if not os.path.isdir(run_path):
                continue

            # Determine config from dirname: baseline_* or prowl_*
            if run_dir_name.startswith("baseline"):
                config_name = "baseline"
            elif run_dir_name.startswith("prowl"):
                config_name = "prowl"
            else:
                continue

            # Find CSV
            csv_files = glob.glob(os.path.join(run_path, "*.csv"))
            if not csv_files:
                continue

            aiperf_data = parse_aiperf_csv(csv_files[0])
            if not aiperf_data:
                continue

            # Extract batch size from dir name if present (default 16 per script)
            bs_match = re.search(r"bs(\d+)", dir_name)
            batch_size = int(bs_match.group(1)) if bs_match else 16

            tp = KNOWN_TP.get(model_name, "")

            rows.append({
                "model": model_name,
                "benchmark": "aiperf_mooncake",
                "nshot": "",
                "tp": tp,
                "num_gpus": tp * 2 if is_disagg and tp else (tp if tp else ""),
                "batch_size": batch_size,
                "mode": "disagg" if is_disagg else "colocated",
                "cuda_graph_enabled": "",
                "cudagraph_mode": "",
                "config": config_name,
                "accuracy": "",
                "accuracy_metric": "",
                "p50_tpot_ms_direct": "",
                "p90_tpot_ms_direct": "",
                "p99_tpot_ms_direct": "",
                "mean_tpot_ms_direct": "",
                "p50_itl_ms_direct": aiperf_data.get("p50_itl_ms_direct", ""),
                "p90_itl_ms_direct": aiperf_data.get("p90_itl_ms_direct", ""),
                "p99_itl_ms_direct": aiperf_data.get("p99_itl_ms_direct", ""),
                "mean_itl_ms_direct": aiperf_data.get("mean_itl_ms_direct", ""),
                "hist_granularity": "",
                "hist_n_buckets": "",
                "p50_tpot_ms_hist_fine": "",
                "p90_tpot_ms_hist_fine": "",
                "p99_tpot_ms_hist_fine": "",
                "mean_tpot_ms_hist_fine": "",
                "p50_tpot_ms_hist_coarse": "",
                "p90_tpot_ms_hist_coarse": "",
                "p99_tpot_ms_hist_coarse": "",
                "mean_tpot_ms_hist_coarse": "",
                "source": f"aiperf:{dir_name}",
            })

    return rows


# ─────────────────────────────────────────────────────────────────────────
#  Compute speedup and accuracy degradation vs baseline
# ─────────────────────────────────────────────────────────────────────────

def _avg_floats(values):
    """Average a list of values, ignoring empty/None entries."""
    nums = []
    for v in values:
        if v not in ("", None):
            try:
                nums.append(float(v))
            except (ValueError, TypeError):
                pass
    return sum(nums) / len(nums) if nums else ""


def compute_comparisons(rows):
    # Build baseline index: group key -> averaged baseline values
    # Multiple baselines for the same group are averaged (repeat runs).
    from collections import defaultdict
    baseline_groups = defaultdict(list)
    for row in rows:
        if is_baseline(row["config"]):
            key = (row["model"], row["benchmark"], row["mode"],
                   row["tp"], row["batch_size"], row["source"])
            baseline_groups[key].append(row)

    baselines = {}
    avg_cols = ["accuracy",
                "p50_tpot_ms_hist_fine", "p90_tpot_ms_hist_fine",
                "p99_tpot_ms_hist_fine", "mean_tpot_ms_hist_fine",
                "p50_tpot_ms_hist_coarse", "p90_tpot_ms_hist_coarse",
                "p99_tpot_ms_hist_coarse", "mean_tpot_ms_hist_coarse",
                "p50_tpot_ms_direct", "p90_tpot_ms_direct",
                "p99_tpot_ms_direct", "mean_tpot_ms_direct",
                "p50_itl_ms_direct", "p90_itl_ms_direct",
                "p99_itl_ms_direct", "mean_itl_ms_direct"]
    for key, group in baseline_groups.items():
        if len(group) == 1:
            baselines[key] = group[0]
        else:
            # Average numeric columns across repeat runs
            averaged = dict(group[0])  # copy first row for non-numeric fields
            for col in avg_cols:
                averaged[col] = _avg_floats([r.get(col) for r in group])
            baselines[key] = averaged

    for row in rows:
        key = (row["model"], row["benchmark"], row["mode"],
               row["tp"], row["batch_size"], row["source"])
        baseline = baselines.get(key)
        row["is_baseline"] = is_baseline(row["config"])

        if baseline and not row["is_baseline"]:
            # Accuracy delta: prowl - baseline (negative = prowl is worse)
            if row["accuracy"] not in ("", None) and baseline["accuracy"] not in ("", None):
                try:
                    row["accuracy_delta"] = (float(row["accuracy"]) - float(baseline["accuracy"])) * 100
                except (ValueError, TypeError):
                    row["accuracy_delta"] = ""
            else:
                row["accuracy_delta"] = ""

            # Speedup for all latency columns: baseline / prowl (>1 = prowl faster)
            for suffix in ("_direct", "_hist_fine", "_hist_coarse"):
                for metric_base in ("p50_tpot_ms", "p90_tpot_ms", "p99_tpot_ms", "mean_tpot_ms",
                                    "p50_itl_ms", "p90_itl_ms", "p99_itl_ms", "mean_itl_ms"):
                    col = f"{metric_base}{suffix}"
                    speedup_col = col.replace("_ms_", "_speedup_")
                    try:
                        b_val = float(baseline[col])
                        p_val = float(row[col])
                        row[speedup_col] = b_val / p_val if p_val > 0 else ""
                    except (ValueError, TypeError, KeyError):
                        row[speedup_col] = ""
        else:
            row["accuracy_delta"] = ""
            for suffix in ("_direct", "_hist_fine", "_hist_coarse"):
                for metric_base in ("p50_tpot", "p90_tpot", "p99_tpot", "mean_tpot",
                                    "p50_itl", "p90_itl", "p99_itl", "mean_itl"):
                    row[f"{metric_base}_speedup{suffix}"] = ""


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────

COLUMNS = [
    # Experiment settings
    "model", "benchmark", "nshot", "tp", "num_gpus", "batch_size", "mode",
    "cuda_graph_enabled", "cudagraph_mode",
    # Config
    "config", "is_baseline",
    # Accuracy
    "accuracy", "accuracy_metric", "accuracy_delta",
    # Speedup summary (right after accuracy for quick reading)
    "p50_tpot_speedup_hist_fine", "mean_tpot_speedup_hist_fine",
    # TPOT direct (from bench_bs*.json)
    "p50_tpot_ms_direct", "p90_tpot_ms_direct", "p99_tpot_ms_direct", "mean_tpot_ms_direct",
    # TPOT histogram — fine granularity (109 buckets, 1ms steps)
    "hist_granularity", "hist_n_buckets",
    "p50_tpot_ms_hist_fine", "p90_tpot_ms_hist_fine", "p99_tpot_ms_hist_fine", "mean_tpot_ms_hist_fine",
    # TPOT histogram — coarse granularity (19 buckets, 10-25ms steps)
    "p50_tpot_ms_hist_coarse", "p90_tpot_ms_hist_coarse", "p99_tpot_ms_hist_coarse", "mean_tpot_ms_hist_coarse",
    # ITL direct (from bench_bs*.json or AIPerf CSV)
    "p50_itl_ms_direct", "p90_itl_ms_direct", "p99_itl_ms_direct", "mean_itl_ms_direct",
    # Remaining speedup columns
    "p90_tpot_speedup_hist_fine", "p99_tpot_speedup_hist_fine",
    "p50_tpot_speedup_direct", "p90_tpot_speedup_direct", "p99_tpot_speedup_direct", "mean_tpot_speedup_direct",
    "p50_itl_speedup_direct", "p90_itl_speedup_direct", "p99_itl_speedup_direct", "mean_itl_speedup_direct",
    "p50_tpot_speedup_hist_coarse", "p90_tpot_speedup_hist_coarse", "p99_tpot_speedup_hist_coarse", "mean_tpot_speedup_hist_coarse",
    # Metadata
    "source",
]


def fmt(val):
    if val is None or val == "":
        return ""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def main():
    parser = argparse.ArgumentParser(description="Gather all Prowl benchmark results into a CSV.")
    parser.add_argument("results_dir", nargs="?",
                        default="/data/vgupta345/prowl_related_data/prowl-open-source/results",
                        help="Root results directory")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV path (default: {results_dir}/all_results.csv)")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_path = args.output or os.path.join(results_dir, "all_results.csv")

    print(f"Scanning: {results_dir}")

    # 1. Parse orchestration logs -> populate KNOWN_TP
    logs_dir = os.path.join(results_dir, "logs")
    disagg_gpu_info = parse_orchestration_logs(logs_dir)

    # 2. Build server_log index -> per-run TP, batch_size, cudagraph
    print("  Indexing server logs...")
    server_log_index = build_server_log_index(results_dir)
    print(f"    Indexed {len(server_log_index)} server log entries")

    # 3. Gather lm-eval results
    print("  Gathering lm-eval results...")
    lm_eval_rows = gather_lm_eval_results(results_dir, server_log_index, disagg_gpu_info)
    print(f"    Found {len(lm_eval_rows)} lm-eval results")

    # 4. Gather batch-size sweep results
    print("  Gathering batch-size sweep results...")
    bs_sweep_rows = gather_bs_sweep_results(results_dir)
    print(f"    Found {len(bs_sweep_rows)} batch-size sweep results")

    # 5. Gather AIPerf trace results
    print("  Gathering AIPerf trace results...")
    aiperf_rows = gather_aiperf_results(results_dir)
    print(f"    Found {len(aiperf_rows)} AIPerf results")

    all_rows = lm_eval_rows + bs_sweep_rows + aiperf_rows

    if not all_rows:
        print("No results found.")
        return

    # 6. Compute comparisons
    print("  Computing comparisons vs baseline...")
    compute_comparisons(all_rows)

    # Summary
    n_colocated_lm = sum(1 for r in all_rows if r["mode"] == "colocated" and r["source"] == "lm-eval")
    n_disagg = sum(1 for r in all_rows if r["mode"] == "disagg" and r["source"] == "lm-eval")
    n_sweep = len(bs_sweep_rows)
    n_aiperf = len(aiperf_rows)
    n_with_acc = sum(1 for r in all_rows if r["accuracy"] not in ("", None))
    n_with_tpot_direct = sum(1 for r in all_rows if r["p50_tpot_ms_direct"] not in ("", None))
    n_with_tpot_hist_fine = sum(1 for r in all_rows if r.get("p50_tpot_ms_hist_fine") not in ("", None))
    n_with_tpot_hist_coarse = sum(1 for r in all_rows if r.get("p50_tpot_ms_hist_coarse") not in ("", None))
    n_with_itl = sum(1 for r in all_rows if r["p50_itl_ms_direct"] not in ("", None))
    n_with_cudagraph = sum(1 for r in all_rows if r["cuda_graph_enabled"] not in ("", None))

    print(f"\n  Summary:")
    print(f"    Total rows:              {len(all_rows)}")
    print(f"    Colocated (lm-eval):     {n_colocated_lm}")
    print(f"    Disaggregated (lm-eval): {n_disagg}")
    print(f"    Batch-size sweep:        {n_sweep}")
    print(f"    AIPerf trace:            {n_aiperf}")
    print(f"    With accuracy:           {n_with_acc}")
    print(f"    With TPOT (direct):      {n_with_tpot_direct}")
    print(f"    With TPOT (hist fine):   {n_with_tpot_hist_fine}")
    print(f"    With TPOT (hist coarse): {n_with_tpot_hist_coarse}")
    print(f"    With ITL (direct):       {n_with_itl}")
    print(f"    With CUDA graph info:    {n_with_cudagraph}")

    # Sort
    all_rows.sort(key=lambda r: (r["model"], r["benchmark"], r["mode"],
                                  str(r["batch_size"]),
                                  not r.get("is_baseline", False),
                                  r["config"]))

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in all_rows:
            writer.writerow([fmt(row.get(col, "")) for col in COLUMNS])

    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
