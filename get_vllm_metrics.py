import sys
import re

def parse_metrics(file_path):
    """Parse Prometheus-formatted metrics file into structured data."""
    metrics = {}
    histograms = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Parse metrics with labels: metric_name{labels} value
            match = re.match(r'^([^{]+)\{(.*?)\}\s+([\d.e+-]+)$', line)
            if match:
                metric_name, labels, value = match.groups()
                
                # Handle histogram buckets
                if metric_name.endswith('_bucket'):
                    hist_name = metric_name[:-len('_bucket')]
                    le_match = re.search(r'le="([^"]+)"', labels)
                    if le_match:
                        bucket_limit = le_match.group(1)
                        if hist_name not in histograms:
                            histograms[hist_name] = {'buckets': [], 'values': []}
                        histograms[hist_name]['buckets'].append(bucket_limit)
                        histograms[hist_name]['values'].append(float(value))
                else:
                    # Store with full label line for finish_reason lookups
                    full_key = f"{metric_name}{{{labels}}}"
                    metrics[full_key] = float(value)
                    # Also store by bare metric name (last one wins, fine for gauges/simple counters)
                    metrics[metric_name] = float(value)
    
    return metrics, histograms

def calculate_percentile(histogram, percentile):
    """Calculate approximate percentile from histogram buckets using linear interpolation."""
    if not histogram['values']:
        return 0
    
    total_count = histogram['values'][-1]  # +Inf bucket
    if total_count == 0:
        return 0
    
    target_count = total_count * (percentile / 100.0)
    
    prev_count = 0.0
    prev_bound = 0.0
    
    for i, count in enumerate(histogram['values']):
        if count >= target_count:
            bucket_str = histogram['buckets'][i]
            if bucket_str == '+Inf':
                return float(histogram['buckets'][i-1]) if i > 0 else 0
            
            upper_bound = float(bucket_str)
            # Linear interpolation within the bucket
            if count == prev_count:
                return upper_bound
            fraction = (target_count - prev_count) / (count - prev_count)
            return prev_bound + fraction * (upper_bound - prev_bound)
        
        prev_count = count
        try:
            prev_bound = float(histogram['buckets'][i])
        except ValueError:
            prev_bound = 0.0
    
    return 0

def get_metric(metrics, *candidates, default=0):
    """Try multiple metric name candidates, return first found."""
    for name in candidates:
        if name in metrics:
            return metrics[name]
    return default

def get_finish_reason_count(metrics, reason):
    """Sum all request_success_total entries matching a given finished_reason."""
    total = 0
    for key, value in metrics.items():
        if 'request_success_total' in key and f'finished_reason="{reason}"' in key:
            total += value
    return total

def compute_metrics(file_path):
    """Compute and display key performance metrics from vLLM metrics file."""
    metrics, histograms = parse_metrics(file_path)
    
    print("=" * 60)
    print("vLLM Performance Metrics Summary")
    print("=" * 60)
    
    # --- Throughput ---
    print("\n📊 THROUGHPUT METRICS:")
    
    gen_tokens = get_metric(metrics,
        "vllm:request_generation_tokens_sum",
        "vllm:generation_tokens_total")
    prompt_tokens = get_metric(metrics, "vllm:prompt_tokens_total")
    decode_time = get_metric(metrics,
        "vllm:request_decode_time_seconds_sum", default=1)
    prefill_time = get_metric(metrics,
        "vllm:request_prefill_time_seconds_sum", default=1)
    total_time = get_metric(metrics,
        "vllm:e2e_request_latency_seconds_sum", default=1)
    
    # TPOT from the per-request histogram sum/count
    tpot_sum = get_metric(metrics,
        "vllm:request_time_per_output_token_seconds_sum",
        "vllm:time_per_output_token_seconds_sum")
    tpot_count = get_metric(metrics,
        "vllm:request_time_per_output_token_seconds_count",
        "vllm:time_per_output_token_seconds_count", default=1)
    
    if tpot_sum > 0 and tpot_count > 0:
        mean_tpot_ms = 1000 * tpot_sum / tpot_count
        print(f"  Mean TPOT:             {mean_tpot_ms:.2f} ms")
    
    # TPOT p50 from histogram
    tpot_hist_key = None
    for k in ['vllm:request_time_per_output_token_seconds',
              'vllm:time_per_output_token_seconds']:
        if k in histograms:
            tpot_hist_key = k
            break
    if tpot_hist_key:
        tpot_p50 = calculate_percentile(histograms[tpot_hist_key], 50)
        tpot_p90 = calculate_percentile(histograms[tpot_hist_key], 90)
        tpot_p99 = calculate_percentile(histograms[tpot_hist_key], 99)
        print(f"  TPOT P50: {tpot_p50*1000:.2f} ms | P90: {tpot_p90*1000:.2f} ms | P99: {tpot_p99*1000:.2f} ms")
    
    gen_throughput = gen_tokens / decode_time if decode_time > 0 else 0
    prefill_throughput = prompt_tokens / prefill_time if prefill_time > 0 else 0
    overall_throughput = (gen_tokens + prompt_tokens) / total_time if total_time > 0 else 0
    
    print(f"  Generation Throughput: {gen_throughput:.2f} tokens/sec")
    print(f"  Prefill Throughput:    {prefill_throughput:.2f} tokens/sec")
    print(f"  Overall Throughput:    {overall_throughput:.2f} tokens/sec")
    
    # --- Request Statistics ---
    print("\n📈 REQUEST STATISTICS:")
    num_running = get_metric(metrics, "vllm:num_requests_running")
    num_waiting = get_metric(metrics, "vllm:num_requests_waiting")
    num_swapped = get_metric(metrics, "vllm:num_requests_swapped")
    
    print(f"  Currently Running: {int(num_running)} requests")
    print(f"  Currently Waiting: {int(num_waiting)} requests")
    if num_swapped > 0:
        print(f"  Currently Swapped: {int(num_swapped)} requests")
    
    stop_count = get_finish_reason_count(metrics, "stop")
    length_count = get_finish_reason_count(metrics, "length")
    abort_count = get_finish_reason_count(metrics, "abort")
    error_count = get_finish_reason_count(metrics, "error")
    total_requests = stop_count + length_count
    
    if total_requests > 0:
        print(f"  Total Completed:   {int(total_requests)} requests")
        print(f"    Stopped (EOS):   {int(stop_count)}")
        print(f"    Max Length:      {int(length_count)}")
        if abort_count > 0:
            print(f"    Aborted:         {int(abort_count)}")
        if error_count > 0:
            print(f"    Errors:          {int(error_count)}")
    
    # --- Token Statistics ---
    print("\n🔢 TOKEN STATISTICS:")
    avg_prompt = prompt_tokens / total_requests if total_requests > 0 else 0
    avg_gen = gen_tokens / total_requests if total_requests > 0 else 0
    
    print(f"  Total Prompt Tokens:      {int(prompt_tokens)}")
    print(f"  Total Generated Tokens:   {int(gen_tokens)}")
    print(f"  Avg Prompt Tokens/Req:    {avg_prompt:.1f}")
    print(f"  Avg Generated Tokens/Req: {avg_gen:.1f}")
    if gen_tokens > 0:
        print(f"  Input/Output Ratio:       {prompt_tokens/gen_tokens:.1f}")
    
    # --- Latency Percentiles ---
    print("\n⏱️  LATENCY PERCENTILES:")
    
    def print_hist_percentiles(label, hist_key, unit_scale=1.0, unit_suffix="s", fmt=".3f"):
        if hist_key in histograms:
            h = histograms[hist_key]
            p50 = calculate_percentile(h, 50) * unit_scale
            p90 = calculate_percentile(h, 90) * unit_scale
            p99 = calculate_percentile(h, 99) * unit_scale
            print(f"  {label}:")
            print(f"    P50: {p50:{fmt}}{unit_suffix} | P90: {p90:{fmt}}{unit_suffix} | P99: {p99:{fmt}}{unit_suffix}")
    
    print_hist_percentiles("Time to First Token",
        "vllm:time_to_first_token_seconds", 1000, "ms", ".1f")
    
    # Inter-token latency (per-token granularity)
    print_hist_percentiles("Inter-Token Latency",
        "vllm:inter_token_latency_seconds", 1000, "ms", ".1f")
    
    # Per-request TPOT
    for k in ['vllm:request_time_per_output_token_seconds',
              'vllm:time_per_output_token_seconds']:
        if k in histograms:
            print_hist_percentiles("Per-Request TPOT", k, 1000, "ms", ".2f")
            break
    
    # E2E latency
    print_hist_percentiles("End-to-End Latency",
        "vllm:e2e_request_latency_seconds", 1.0, "s", ".2f")
    
    # Queue time
    print_hist_percentiles("Queue Time",
        "vllm:request_queue_time_seconds", 1000, "ms", ".1f")
    
    # Prefill time
    print_hist_percentiles("Prefill Time",
        "vllm:request_prefill_time_seconds", 1000, "ms", ".1f")
    
    # Decode time
    print_hist_percentiles("Decode Time",
        "vllm:request_decode_time_seconds", 1.0, "s", ".2f")
    
    # --- Cache Utilization ---
    print("\n💾 CACHE UTILIZATION:")
    kv_cache = get_metric(metrics,
        "vllm:gpu_cache_usage_perc",
        "vllm:kv_cache_usage_perc")
    cpu_cache = get_metric(metrics, "vllm:cpu_cache_usage_perc")
    preemptions = get_metric(metrics, "vllm:num_preemptions_total")
    
    print(f"  KV Cache Usage:    {kv_cache*100:.1f}%")
    if cpu_cache > 0:
        print(f"  CPU Cache Usage:   {cpu_cache*100:.1f}%")
    print(f"  Preemptions:       {int(preemptions)}")
    
    # Prefix cache hit rate
    prefix_queries = get_metric(metrics, "vllm:prefix_cache_queries_total")
    prefix_hits = get_metric(metrics, "vllm:prefix_cache_hits_total")
    if prefix_queries > 0:
        hit_rate = prefix_hits / prefix_queries * 100
        print(f"  Prefix Cache Hits: {int(prefix_hits)}/{int(prefix_queries)} ({hit_rate:.1f}%)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_vllm_metrics.py <metrics_file>")
        sys.exit(1)
    
    compute_metrics(sys.argv[1])