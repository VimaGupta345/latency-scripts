# Sequential Benchmark Run Guide

## Overview

Run all 4 models x 4 benchmarks x 2 configs = 32 runs sequentially on a single machine with no GPU contention. Optionally run twice (with and without the bs=1 TPOT filter) to measure filter impact.

## Models & GPU Requirements

| Model | TP Size | GPUs | Port | Baseline Config | Prowl Config |
|-------|---------|------|------|----------------|--------------|
| DeepSeek-Coder-V2 (236B) | 4 | 0,1,2,3 | 8020 | `config_do_nothing.json` | `quant_alpha1.125_beta2_optimized.json` |
| Mixtral-8x7B | 2 | 0,1 | 8040 | `do_nothing.json` | `quant_alpha0.7_beta1_optimized.json` |
| Qwen2-57B-A14B | 2 | 0,1 | 8030 | `qwen_do-nothing.json` | `quant_alpha3_beta4_optimized.json` |
| Qwen3-30B-A3B | 1 | 0 | 8019 | `qwen_do-nothing.json` | `quant_alpha3_beta2_optimized.json` |

## Benchmarks

Each model runs: `humaneval` (n=164), `mbpp` (n=250), `gsm8k` (n=250), `minerva_math_algebra` (n=250).

## How to Run

### Pass 1: With bs=1 filter (default)

```zsh
cd /data/vgupta345/prowl_related_data/prowl-open-source/latency-scripts
zsh run_all_sequential.sh
```

### Flipping the bs=1 Filter

The filter is in `prowl/vllm/v1/metrics/loggers.py` around line 564-569.

**With filter (default):**
```python
        # Only record TPOT when batch size > 1. At bs=1, Prowl's
        # optimization does not apply, so including these observations
        # would dilute the measured throughput.
        if scheduler_stats is None or scheduler_stats.num_running_reqs > 1:
            for tpot in iteration_stats.time_per_output_tokens_iter:
                self.histogram_time_per_output_token[engine_idx].observe(tpot)
```

**Without filter (record all TPOT):**
```python
        # Record all TPOT observations (bs=1 filter disabled)
        for tpot in iteration_stats.time_per_output_tokens_iter:
            self.histogram_time_per_output_token[engine_idx].observe(tpot)
```

Note: The filter ONLY affects the TPOT histogram. It does NOT affect:
- `request_decode_time_seconds` (used for throughput computation)
- `request_generation_tokens` (token counts)
- `e2e_request_latency_seconds` (end-to-end latency)
- `iteration_prompt_tokens` / `iteration_generation_tokens` (I3 histograms)

### Pass 2: Without bs=1 filter

1. Edit `prowl/vllm/v1/metrics/loggers.py` — remove the `if` guard as shown above
2. Run again:
```zsh
cd /data/vgupta345/prowl_related_data/prowl-open-source/latency-scripts
zsh run_all_sequential.sh
```
3. Restore the filter after the run if desired

## Collecting Results

### Where results land

Each run produces a `.metrics` file at:
```
results/<ModelName>/<benchmark>/adv_fp16_<model>_<benchmark>_port<port>_n<limit>_conf_<config>_<timestamp>.metrics
```

Example:
```
results/Qwen2-57B-A14B-Instruct/humaneval/adv_fp16_qwen_humaneval_port8030_n164_conf_qwen_do-nothing_20260317-220317.metrics
```

Logs go to `results/logs/all_sequential_<timestamp>.log`.

### Collating speedup results

From `latency-scripts/`:

```bash
source /data/vgupta345/prowl_related_data/prowl-open-source/prowl/.venv/bin/activate
python3 /tmp/collate_final2.py
```

This script:
- Finds the newest fine-grained (110 TPOT buckets) `.metrics` file for each model/benchmark/config
- Computes P25/P50/P75/P90/P99 TPOT speedup (baseline/prowl, >1 = faster)
- Computes throughput speedup as `gen_tokens / decode_time` (matching `get_vllm_metrics.py`)
- Computes E2E request latency speedup

### Quick single-file analysis

```bash
python3 get_vllm_metrics.py <path_to_metrics_file>
```

### Verifying instrumentation

To confirm a `.metrics` file has the fine-grained buckets and I3 histograms:
```bash
# Should show ~110 lines (100 fine + 10 coarse)
grep -c 'time_per_output_token_seconds_bucket' <metrics_file>

# Should show 14+ lines
grep -c 'iteration_prompt_tokens_bucket' <metrics_file>
```

## Instrumentation Details

Two changes were made to `prowl/vllm/v1/metrics/loggers.py`:

### 1. Fine-grained TPOT buckets (line ~400)

Uniform 1ms buckets from 1-100ms, then coarse tail buckets. Enables precise percentile estimates in the decode latency range.

```python
tpot_buckets = [i * 0.001 for i in range(1, 101)]  # 1ms to 100ms
tpot_buckets += [0.15, 0.2, 0.3, 0.5, 1.0, 5.0, 10.0, 20.0, 80.0]
```

### 2. I3 iteration histograms (line ~338)

Per-engine-step histograms for prompt and generation tokens. Used to identify chunked prefill steps.

- `vllm:iteration_prompt_tokens` — steps with >1 prompt token are prefill-interleaved
- `vllm:iteration_generation_tokens` — decode batch size distribution

## Expected Runtime

~5-10 min per benchmark run. 32 runs per pass = ~3-5 hours total.
