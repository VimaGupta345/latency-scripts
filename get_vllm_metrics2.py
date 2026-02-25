import sys
import re

def parse_metrics(file_path):
    """Parse Prometheus-formatted metrics file into structured data."""
    metrics = {}
    histograms = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            # Parse simple metrics - extract just the metric name without labels
            match = re.match(r'^([^{]+)\{.*?\}\s+([\d.e+-]+)$', line.strip())
            if match:
                full_key, value = match.groups()
                
                # Handle histogram buckets specially
                if '_bucket' in full_key:
                    # Extract histogram name and bucket limit
                    hist_match = re.match(r'^(.*?)_bucket$', full_key)
                    if hist_match:
                        hist_name = hist_match.group(1)
                        # Extract le value from labels
                        le_match = re.search(r'le="([^"]+)"', line)
                        if le_match:
                            bucket_limit = le_match.group(1)
                            if hist_name not in histograms:
                                histograms[hist_name] = {'buckets': [], 'values': []}
                            histograms[hist_name]['buckets'].append(bucket_limit)
                            histograms[hist_name]['values'].append(float(value))
                else:
                    # Store with simplified key (metric name only)
                    metrics[full_key] = float(value)
                    
                    # Also store the full line for finish_reason parsing
                    if 'finished_reason' in line:
                        metrics[line.split()[0]] = float(value)
    
    return metrics, histograms

def calculate_percentile(histogram, percentile):
    """Calculate approximate percentile from histogram buckets."""
    if not histogram['values']:
        return 0
    
    total_count = histogram['values'][-1]  # Last bucket is +Inf
    if total_count == 0:
        return 0
    
    target_count = total_count * (percentile / 100.0)
    
    for i, count in enumerate(histogram['values']):
        if count >= target_count:
            if histogram['buckets'][i] == '+Inf':
                # Return the previous bucket if we hit +Inf
                return float(histogram['buckets'][i-1]) if i > 0 else 0
            try:
                return float(histogram['buckets'][i])
            except ValueError:
                return 0
    return 0

def compute_metrics(file_path):
    """Compute and display key performance metrics from vLLM metrics file."""
    metrics, histograms = parse_metrics(file_path)
    
    print("=" * 60)
    print("vLLM Performance Metrics Summary")
    print("=" * 60)
    
    # 1. Throughput Metrics
    print("\n📊 THROUGHPUT METRICS:")
    gen_tokens = metrics.get("vllm:request_generation_tokens_sum", 0)
    prompt_tokens = metrics.get("vllm:prompt_tokens_total", 0)
    decode_time = metrics.get("vllm:request_decode_time_seconds_sum", 0)
    prefill_time = metrics.get("vllm:request_prefill_time_seconds_sum", 0)
    total_time = metrics.get("vllm:e2e_request_latency_seconds_sum", 0)

    # vLLM metric naming varies by version.
    tpot_seconds_sum = metrics.get(
        "vllm:request_time_per_output_token_seconds_sum",
        metrics.get("vllm:time_per_output_token_seconds_sum", 0),
    )
    total_output_tokens_sum = metrics.get(
        "vllm:request_time_per_output_token_seconds_count",
        metrics.get("vllm:time_per_output_token_seconds_count", 0),
    )

    # Keep one TPOT output line while supporting multiple vLLM metric variants.
    if total_output_tokens_sum > 0:
        tpot_ms = tpot_seconds_sum / total_output_tokens_sum * 1000
        print(f"  TPOT: {tpot_ms:.2f} ms")
    else:
        itl_sum = metrics.get("vllm:inter_token_latency_seconds_sum", 0)
        itl_count = metrics.get("vllm:inter_token_latency_seconds_count", 0)
        if itl_count > 0:
            tpot_ms = itl_sum / itl_count * 1000
            print(f"  TPOT: {tpot_ms:.2f} ms")
        elif gen_tokens > 0 and decode_time > 0:
            tpot_ms = decode_time / gen_tokens * 1000
            print(f"  TPOT: {tpot_ms:.2f} ms")
        else:
            print("  TPOT: n/a (no output-token counters found)")
    
    gen_throughput = gen_tokens / decode_time if decode_time > 0 else 0
    prefill_throughput = prompt_tokens / prefill_time if prefill_time > 0 else 0
    overall_throughput = (gen_tokens + prompt_tokens) / total_time if total_time > 0 else 0
    
    print(f"  Generation Throughput: {gen_throughput:.2f} tokens/sec")
    print(f"  Prefill Throughput:    {prefill_throughput:.2f} tokens/sec")
    print(f"  Overall Throughput:    {overall_throughput:.2f} tokens/sec")
    
    # 2. Request Statistics
    print("\n📈 REQUEST STATISTICS:")
    num_running = metrics.get("vllm:num_requests_running", 0)
    num_waiting = metrics.get("vllm:num_requests_waiting", 0)
    num_swapped = metrics.get("vllm:num_requests_swapped", 0)
    
    print(f"  Currently Running: {int(num_running)} requests")
    print(f"  Currently Waiting: {int(num_waiting)} requests")
    print(f"  Currently Swapped: {int(num_swapped)} requests")
    
    # Get finish reasons - look for all request_success_total entries
    stop_count = 0
    length_count = 0
    for key, value in metrics.items():
        if 'request_success_total' in key and 'finished_reason="stop"' in key:
            stop_count += value
        elif 'request_success_total' in key and 'finished_reason="length"' in key:
            length_count += value
    
    total_requests = stop_count + length_count
    if total_requests > 0:
        print(f"  Total Completed:  {int(total_requests)} requests")
        print(f"  Finish Reasons:")
        print(f"    - Stopped (EOS):     {int(stop_count)} requests")
        print(f"    - Max Length:        {int(length_count)} requests")
    
    # 3. Token Statistics
    print("\n🔢 TOKEN STATISTICS:")
    avg_prompt_tokens = prompt_tokens / total_requests if total_requests > 0 else 0
    avg_gen_tokens = gen_tokens / total_requests if total_requests > 0 else 0
    
    print(f"  Total Prompt Tokens:     {int(prompt_tokens)}")
    print(f"  Total Generated Tokens:  {int(gen_tokens)}")
    print(f"  Avg Prompt Tokens/Req:   {avg_prompt_tokens:.1f}")
    print(f"  Avg Generated Tokens/Req: {avg_gen_tokens:.1f}")
    
    # 4. Latency Percentiles
    print("\n⏱️  LATENCY PERCENTILES:")
    
    # Time to First Token
    if 'vllm:time_to_first_token_seconds' in histograms:
        ttft_hist = histograms['vllm:time_to_first_token_seconds']
        ttft_p50 = calculate_percentile(ttft_hist, 50)
        ttft_p90 = calculate_percentile(ttft_hist, 90)
        ttft_p99 = calculate_percentile(ttft_hist, 99)
        print(f"  Time to First Token:")
        print(f"    P50: {ttft_p50:.3f}s | P90: {ttft_p90:.3f}s | P99: {ttft_p99:.3f}s")
    
    # Inter-token Latency
    tpot_hist = histograms.get(
        'vllm:request_time_per_output_token_seconds',
        histograms.get('vllm:time_per_output_token_seconds')
    )
    if tpot_hist:
        tpot_p50 = calculate_percentile(tpot_hist, 50)
        tpot_p90 = calculate_percentile(tpot_hist, 90)
        tpot_p99 = calculate_percentile(tpot_hist, 99)
        print(f"  Inter-token Latency:")
        print(f"    P50: {tpot_p50*1000:.1f}ms | P90: {tpot_p90*1000:.1f}ms | P99: {tpot_p99*1000:.1f}ms")
    
    # End-to-end Latency
    if 'vllm:e2e_request_latency_seconds' in histograms:
        e2e_hist = histograms['vllm:e2e_request_latency_seconds']
        e2e_p50 = calculate_percentile(e2e_hist, 50)
        e2e_p90 = calculate_percentile(e2e_hist, 90)
        e2e_p99 = calculate_percentile(e2e_hist, 99)
        print(f"  End-to-End Latency:")
        print(f"    P50: {e2e_p50:.2f}s | P90: {e2e_p90:.2f}s | P99: {e2e_p99:.2f}s")
    
    # 5. Cache Utilization
    print("\n💾 CACHE UTILIZATION:")
    gpu_cache = metrics.get("vllm:gpu_cache_usage_perc", 0)
    cpu_cache = metrics.get("vllm:cpu_cache_usage_perc", 0)
    preemptions = metrics.get("vllm:num_preemptions_total", 0)
    
    print(f"  GPU Cache Usage: {gpu_cache*100:.1f}%")
    print(f"  CPU Cache Usage: {cpu_cache*100:.1f}%")
    print(f"  Total Preemptions: {int(preemptions)}")
    
    print("\n" + "=" * 60)
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    compute_metrics(file_path)
