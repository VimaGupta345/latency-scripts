# PROWL Latency & Quality Benchmarking Scripts

Benchmarking toolkit for measuring quality-latency tradeoffs of **PROWL/Midas** expert activation optimization on Mixture-of-Experts (MoE) LLMs served via a custom vLLM fork.

## Overview

The workflow is:

1. **Launch** a vLLM OpenAI-compatible API server with a PROWL configuration
2. **Run** lm-eval-harness benchmarks against the server
3. **Collect** Prometheus-formatted metrics from vLLM's `/metrics` endpoint
4. **Analyze** accuracy vs. throughput/TPOT tradeoffs across configurations

### Supported Models

| Model | TP Size | Config Directory |
|-------|---------|-----------------|
| Mixtral-8x7B-Instruct-v0.1 | 1-2 | `configs/mixtral/` |
| Qwen2-57B-A14B-Instruct | 2 | `configs/qwen/` |
| Qwen3-235B-A22B-Instruct-2507 | 4 | `configs/qwen3/` |
| DeepSeek-V2-Lite-Chat | 1 | `configs/deepseek/` |
| Llama-4-Scout-17B-16E-Instruct | 4 | `configs/llama4/` |

### Supported Benchmarks

GSM8K, HumanEval, MBPP, Minerva Math Algebra, TruthfulQA, TriviaQA, MT-Bench, HotPotQA, XSUM, CNN/DailyMail, CoQA, NarrativeQA, SQuAD, SQuADv2, MMMU, MathVista.

## File Structure

### Core Python Scripts

| File | Purpose |
|------|---------|
| `lm_eval_online_serve.py` | **Main orchestrator.** Starts a vLLM server, runs lm-eval benchmarks, collects metrics, shuts down the server. Supports single-benchmark and mixed (concurrent multi-benchmark) modes. |
| `get_vllm_metrics.py` | Parses a Prometheus-formatted vLLM `.metrics` file and prints a human-readable summary (throughput, TPOT percentiles, cache utilization, request stats). |
| `analyze_all_benchmark_results.py` | Aggregates results across models and configs. Produces comparison tables showing accuracy drop vs. speedup relative to a "do_nothing" baseline. Exports to CSV. |
| `to_csv.py` | Parses batched metrics output (from piped shell commands) into a CSV. Handles both full token-stats format and TPOT-only format. |
| `get_length.py` | Earlier version of `to_csv.py` (without TPOT support). |
| `run_qwen3_offline.py` | Quick offline (no server) vLLM smoke test for Qwen3-235B. |

### Shell Scripts — Server Launchers

| File | Purpose |
|------|---------|
| `online_serving_ngram.sh` | Launches a vLLM server on port 8000 with ngram speculative decoding. If `K=0`, runs baseline (no speculation). Configures PROWL/Midas via environment variables. |
| `online_serving_ngram_port.sh` | Same as above but with configurable port, batch size, and `TP_SIZE` override from environment. |

### Shell Scripts — Benchmark Runners

| File | Purpose |
|------|---------|
| `run_mixtral_adv_configurable.sh` | **Generic benchmark runner** (works for all models despite the name). Takes port, model name, model path, benchmark, limit, config file, TP size, and batch size as arguments. |
| `run_mixtral_adv_mini_configurable.sh` | Simplified version — runs fewer benchmark/config combos on a specified port. |
| `run_mixtral_adv.sh` | Hardcoded Mixtral runner (older version, uses fixed paths). |
| `run_mixtral_adv_with_server.sh` | Runs benchmarks against an already-running server (no server launch). |
| `run_mixtral_multi_exp.sh` | Runs multiple experiments with different sample limits. |
| `debug_mixtral_gsm8k_100.sh` | Debug script: Mixtral on GSM8K with a single config (beta 0.7). |

### Shell Scripts — Model-Specific Suites

These wrap `run_mixtral_adv_configurable.sh` with model-specific settings:

| File | Model | Benchmarks |
|------|-------|------------|
| `run_mixtral_full_benchmarks.sh` | Mixtral-8x7B | gsm8k, minerva_math_algebra, humaneval |
| `run_mixtral_partial_benchmarks.sh` | Mixtral-8x7B | gsm8k |
| `run_qwen_full_benchmarks.sh` | Qwen2-57B | humaneval, gsm8k, minerva_math_algebra |
| `run_qwen_partial_benchmarks.sh` | Qwen2-57B | humaneval, gsm8k, mbpp, minerva_math_algebra |
| `run_qwen3_full_benchmarks.sh` | Qwen3-235B | humaneval, minerva_math_algebra |
| `run_qwen_sweep_batchsz.sh` | Qwen2-57B | humaneval (sweeps batch sizes) |
| `run_deepseek_full_benchmarks.sh` | DeepSeek-V2-Lite | hotpotqa, xsum |
| `run_deepseek_partial_benchmarks.sh` | DeepSeek-V2-Lite | gsm8k |
| `run_llama4_full_benchmarks.sh` | Llama-4-Scout | mmmu |

### Other

| File | Purpose |
|------|---------|
| `metrics.txt` | Debug artifact (47 empty JSON objects). |

## Usage

### Prerequisites

- Custom vLLM fork with PROWL/Midas support (provides `--mixtral_config_file` flag)
- `lm-eval` (lm-eval-harness)
- PROWL config files in `$TMP_HOME/prowl/configs/`
- GPU access with CUDA

### Running a Single Benchmark

```bash
# Generic: launch server + run benchmark + collect metrics
python lm_eval_online_serve.py \
    -m "Qwen/Qwen2-57B-A14B-Instruct" \
    -o "my_experiment" \
    -b gsm8k \
    -k 0 \
    -t 300 \
    -cf "$TMP_HOME/prowl/configs/qwen/qwen_do-nothing.json" \
    -sa "localhost:8000"
```

### Running a Full Benchmark Suite

```bash
# Set TMP_HOME for config file resolution
export TMP_HOME=/var/tmp/jae

# Run all Qwen benchmarks
CUDA_VISIBLE_DEVICES=2,3 ./run_qwen_full_benchmarks.sh
```

### Using the Configurable Runner Directly

```bash
# Args: <port> <model_name> <model_path> <benchmark> <limit> [config_file] [tp_size] [batch_size]
CUDA_VISIBLE_DEVICES=0,1 ./run_mixtral_adv_configurable.sh \
    8000 qwen "Qwen/Qwen2-57B-A14B-Instruct" gsm8k 250 \
    "$TMP_HOME/prowl/configs/qwen/qwen_do-nothing.json" 2
```

### Analyzing Results

```bash
# Parse a single metrics file
python get_vllm_metrics.py /path/to/results.metrics

# Aggregate all results into comparison tables
python analyze_all_benchmark_results.py

# Convert batched metrics output to CSV
python to_csv.py -i metrics_dump.txt -o results.csv
```

## Key Concepts

### PROWL/Midas Configuration

PROWL controls expert activation in MoE models. It is configured via:

- **JSON config files**: Contain alpha/beta parameters, quantization settings, and optimization flags.
  - `do_nothing.json` — Baseline (no optimization, all experts active).
  - `advanced_alpha0_beta0.7.json` — Aggressive pruning (fewer experts, faster but potentially lower quality).
  - `quant_alpha1_optimized.json` — Quantization-aware configuration.
- **Environment variables** (set by the serving scripts):
  - `MIDAS_ENABLE` — Enable/disable the optimization.
  - `MIDAS_MAX_EXPERTS` — Maximum number of active experts per token.
  - `MIDAS_CONF_THRES` — Confidence threshold for expert selection.
  - `MIDAS_INFLIGHT_TOKS` — Number of inflight tokens (K+1).
  - `MIDAS_CASD_ENABLE` — Enable context-aware speculative decoding.

### Metrics Collected

From vLLM's Prometheus `/metrics` endpoint:
- **TPOT** (Time Per Output Token) — p50, p90, p99 percentiles
- **TTFT** (Time To First Token)
- **Generation throughput** (tokens/sec)
- **Prefill throughput**
- **End-to-end request latency**
- **GPU/CPU cache utilization**
- **Request completion stats** (stop vs. max-length)

### Output Directory Structure

```
/var/tmp/jae/stats/
  quality/<benchmark>/<spec_decode>/<model_name>/   # lm-eval results + metrics
  perf/spec_decode/ngram/<model_name>/              # vLLM server logs
```

## Known Issues

1. **Duplicate `extra_args`** in `lm_eval_online_serve.py:249` — the lm-eval command appends `{extra_args}` twice.
2. **`args.server_address` scoping bug** in `lm_eval_online_serve.py:284` — mt_bench codepath references `args.server_address` but `args` is not in scope; should use the `server_address` parameter.
3. **`get_length.py` is a near-duplicate of `to_csv.py`** with a misleading filename.
4. **Hardcoded user paths** in several older scripts (`/nethome/vgupta345/`, `/home/asaxena317/`). Newer scripts use `$TMP_HOME`.
5. **Thread safety** — `thread_processes` dict in mixed benchmark mode is mutated from multiple threads without synchronization.
6. **`LIMIT_PERCENT` variable** is defined but never used in model-specific benchmark scripts.
7. **Misleading comments** — `run_llama4_full_benchmarks.sh` has "Qwen" in its comments.
