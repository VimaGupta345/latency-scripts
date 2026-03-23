#!/usr/bin/env python3
"""Compute P50/P90/P99 from Prometheus histogram buckets in .metrics files."""

import sys
import re


def parse_histogram(filepath, metric_prefix):
    """Extract histogram buckets, count, and sum for a given metric prefix."""
    buckets = []
    count = None
    total = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{metric_prefix}_bucket{{"):
                le_match = re.search(r'le="([^"]+)"', line)
                val = float(line.split()[-1])
                le = le_match.group(1)
                if le == "+Inf":
                    le = float('inf')
                else:
                    le = float(le)
                buckets.append((le, val))
            elif line.startswith(f"{metric_prefix}_count{{"):
                count = float(line.split()[-1])
            elif line.startswith(f"{metric_prefix}_sum{{"):
                total = float(line.split()[-1])
    return buckets, count, total


def percentile_from_buckets(buckets, count, p):
    """Linearly interpolate within histogram buckets to estimate percentile."""
    target = count * p
    prev_le = 0.0
    prev_count = 0.0
    for le, cum_count in buckets:
        if le == float('inf'):
            return prev_le  # fallback
        if cum_count >= target:
            # Linear interpolation within this bucket
            bucket_count = cum_count - prev_count
            if bucket_count == 0:
                return le
            fraction = (target - prev_count) / bucket_count
            return prev_le + fraction * (le - prev_le)
        prev_le = le
        prev_count = cum_count
    return prev_le


def analyze(filepath, label):
    metrics = [
        ("vllm:time_per_output_token_seconds", "TPOT (per token)"),
        ("vllm:e2e_request_latency_seconds", "E2E Latency (per req)"),
        ("vllm:request_decode_time_seconds", "Decode Time (per req)"),
        ("vllm:request_prefill_time_seconds", "Prefill Time (per req)"),
        ("vllm:time_to_first_token_seconds", "TTFT (per req)"),
        ("vllm:request_queue_time_seconds", "Queue Time (per req)"),
        ("vllm:request_inference_time_seconds", "Inference Time (per req)"),
    ]

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  File: {filepath.split('/')[-1]}")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Count':>8} {'Mean':>10} {'P50':>10} {'P90':>10} {'P99':>10}")
    print("-" * 73)

    results = {}
    for prefix, name in metrics:
        buckets, count, total = parse_histogram(filepath, prefix)
        if not buckets or count is None or count == 0:
            continue
        mean = total / count
        p50 = percentile_from_buckets(buckets, count, 0.50)
        p90 = percentile_from_buckets(buckets, count, 0.90)
        p99 = percentile_from_buckets(buckets, count, 0.99)

        # Format: use ms for small values, s for large
        if mean < 1.0:
            fmt = lambda v: f"{v*1000:.1f}ms"
        else:
            fmt = lambda v: f"{v:.2f}s"

        print(f"{name:<25} {count:>8.0f} {fmt(mean):>10} {fmt(p50):>10} {fmt(p90):>10} {fmt(p99):>10}")
        results[prefix] = {"count": count, "sum": total, "mean": mean, "p50": p50, "p90": p90, "p99": p99}

    return results


def main():
    baseline_file = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/vgupta345/prowl_related_data/prowl-open-source/results/DeepSeek-Coder-V2-Instruct/humaneval/adv_fp16_deepseek_v2_humaneval_port8020_n164_conf_config_do_nothing_20260316-183713.metrics"
    prowl_file = sys.argv[2] if len(sys.argv) > 2 else \
        "/data/vgupta345/prowl_related_data/prowl-open-source/results/DeepSeek-Coder-V2-Instruct/humaneval/adv_fp16_deepseek_v2_humaneval_port8020_n164_conf_quant_alpha1.175_beta2_optimized_20260316-185640.metrics"

    baseline_results = analyze(baseline_file, "BASELINE (do-nothing)")
    prowl_results = analyze(prowl_file, "PROWL (α1.175 β2)")

    # Comparison table
    print(f"\n{'='*80}")
    print(f"  COMPARISON: Prowl vs Baseline")
    print(f"{'='*80}")

    metrics_to_compare = [
        ("vllm:time_per_output_token_seconds", "TPOT"),
        ("vllm:e2e_request_latency_seconds", "E2E Latency"),
        ("vllm:request_decode_time_seconds", "Decode Time"),
        ("vllm:request_prefill_time_seconds", "Prefill Time"),
        ("vllm:time_to_first_token_seconds", "TTFT"),
    ]

    print(f"{'Metric':<20} {'Mean Speedup':>14} {'P50 Speedup':>14} {'P90 Speedup':>14} {'P99 Speedup':>14}")
    print("-" * 76)

    for prefix, name in metrics_to_compare:
        if prefix in baseline_results and prefix in prowl_results:
            b = baseline_results[prefix]
            p = prowl_results[prefix]
            mean_sp = b["mean"] / p["mean"] if p["mean"] > 0 else float('inf')
            p50_sp = b["p50"] / p["p50"] if p["p50"] > 0 else float('inf')
            p90_sp = b["p90"] / p["p90"] if p["p90"] > 0 else float('inf')
            p99_sp = b["p99"] / p["p99"] if p["p99"] > 0 else float('inf')
            print(f"{name:<20} {mean_sp:>13.3f}x {p50_sp:>13.3f}x {p90_sp:>13.3f}x {p99_sp:>13.3f}x")


if __name__ == "__main__":
    main()
