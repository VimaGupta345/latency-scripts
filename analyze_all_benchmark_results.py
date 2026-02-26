#!/usr/bin/env python3

import os
import re
import json
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

def parse_metrics_file(file_path):
    """Parse Prometheus-formatted metrics file to extract TPOT and throughput data."""
    metrics = {}
    histograms = {}
    
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r'^([^{]+)\{.*?\}\s+([\d.e+-]+)$', line.strip())
            if match:
                full_key, value = match.groups()
                
                if '_bucket' in full_key:
                    hist_match = re.match(r'^(.*?)_bucket$', full_key)
                    if hist_match:
                        hist_name = hist_match.group(1)
                        le_match = re.search(r'le="([^"]+)"', line)
                        if le_match:
                            bucket_limit = le_match.group(1)
                            if hist_name not in histograms:
                                histograms[hist_name] = {'buckets': [], 'values': []}
                            histograms[hist_name]['buckets'].append(bucket_limit)
                            histograms[hist_name]['values'].append(float(value))
                else:
                    metrics[full_key] = float(value)
    
    # Calculate TPOT percentiles
    tpot_data = {}
    if 'vllm:time_per_output_token_seconds' in histograms:
        tpot_hist = histograms['vllm:time_per_output_token_seconds']
        tpot_data['tpot_p50_ms'] = calculate_percentile(tpot_hist, 50) * 1000
        tpot_data['tpot_p90_ms'] = calculate_percentile(tpot_hist, 90) * 1000
        tpot_data['tpot_p99_ms'] = calculate_percentile(tpot_hist, 99) * 1000
    
    # Get throughput metrics and calculate average TPOT
    gen_tokens = metrics.get("vllm:request_generation_tokens_sum", 0)
    decode_time = metrics.get("vllm:request_decode_time_seconds_sum", 1)
    tpot_data['gen_throughput'] = gen_tokens / decode_time if decode_time > 0 else 0
    tpot_data['avg_tpot_ms'] = (decode_time / gen_tokens * 1000) if gen_tokens > 0 else 0
    tpot_data['total_gen_tokens'] = gen_tokens
    tpot_data['total_decode_time'] = decode_time
    
    return tpot_data

def calculate_percentile(histogram, percentile):
    """Calculate approximate percentile from histogram buckets."""
    if not histogram['values']:
        return 0
    
    total_count = histogram['values'][-1]
    if total_count == 0:
        return 0
    
    target_count = total_count * (percentile / 100.0)
    
    for i, count in enumerate(histogram['values']):
        if count >= target_count:
            if histogram['buckets'][i] == '+Inf':
                return float(histogram['buckets'][i-1]) if i > 0 else 0
            try:
                return float(histogram['buckets'][i])
            except ValueError:
                return 0
    return 0

def parse_accuracy_from_log(log_path):
    """Extract accuracy score from the log file."""
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Look for the exact_match value in the results table (for most benchmarks)
    pattern = r'exact_match\|↑?\s*\|\s*([\d.]+)\|'
    matches = re.findall(pattern, content)
    
    if matches:
        # Return the first match (flexible-extract)
        return float(matches[0])
    
    # For humaneval and mbpp, look for pass@1 or pass_at_1
    pattern = r'pass[@_](?:at_)?1\|[↑\s]*\|\s*([\d.]+)\|'
    matches = re.findall(pattern, content)
    
    if matches:
        return float(matches[0])
    
    return None

def get_config_name(file_path):
    """Extract configuration name from file path."""
    # More comprehensive config extraction
    if 'do_nothing' in file_path or 'do-nothing' in file_path:
        return 'do_nothing'
    elif 'beta1.25' in file_path or 'beta1_25' in file_path:
        return 'beta1.25'
    elif 'beta0.7' in file_path or 'beta0_7' in file_path:
        return 'beta0.7'
    elif 'beta0.9' in file_path or 'beta0_9' in file_path:
        return 'beta0.9'
    elif 'beta1.0' in file_path or 'beta1_0' in file_path:
        return 'beta1.0'
    elif 'beta1.1' in file_path or 'beta1_1' in file_path:
        return 'beta1.1'
    elif 'beta2' in file_path:
        return 'beta2.0'
    elif 'config_simple' in file_path:
        return 'simple'
    elif 'config_advanced' in file_path:
        return 'advanced'
    elif 'config_do_nothing' in file_path:
        return 'do_nothing'
    else:
        # Extract config name from path
        match = re.search(r'conf_([^/\.]+)', file_path)
        if match:
            return match.group(1)
    return 'unknown'

def collect_all_benchmark_data():
    """Collect all benchmark data (n500 for most, n164 for humaneval)."""
    base_path = Path('/nethome/jkim3934/stats/quality')
    
    results = []
    
    # Find all log files with n500 or n164
    for log_file in base_path.glob('*/ngram/**/*.log'):
        log_path = str(log_file)
        
        # Check if it's n500 or n164 (humaneval)
        if 'n500' not in log_path and 'n164' not in log_path:
            continue
        
        # Extract information from path
        parts = log_path.split('/')
        benchmark = parts[5]  # e.g., 'gsm8k', 'mbpp', etc.
        model = parts[7]  # e.g., 'Mixtral-8x7B-Instruct-v0.1'
        
        # Get config name
        config = get_config_name(log_path)
        
        # Get corresponding metrics file
        metrics_path = log_path.replace('.log', '.metrics')
        
        # Parse accuracy
        accuracy = parse_accuracy_from_log(log_path)
        
        # Parse metrics
        metrics_data = parse_metrics_file(metrics_path)
        
        if accuracy is not None and metrics_data is not None:
            result = {
                'model': model,
                'benchmark': benchmark,
                'config': config,
                'accuracy': accuracy,
                **metrics_data
            }
            results.append(result)
            print(f"Processed: {model} / {benchmark} / {config}")
    
    return results

def create_comprehensive_comparison(results):
    """Create comprehensive comparison tables."""
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No data found!")
        return
    
    # Sort for consistent ordering
    model_order = ['Mixtral-8x7B-Instruct-v0.1', 'Qwen2-57B-A14B-Instruct', 'DeepSeek-V2-Lite-Chat']
    benchmark_order = ['humaneval', 'gsm8k', 'mbpp', 'minerva_math_algebra']
    
    for model in model_order:
        if model not in df['model'].unique():
            continue
            
        print(f"\n{'='*100}")
        print(f"MODEL: {model}")
        print(f"{'='*100}")
        
        model_df = df[df['model'] == model]
        
        for benchmark in benchmark_order:
            if benchmark not in model_df['benchmark'].unique():
                continue
                
            bench_df = model_df[model_df['benchmark'] == benchmark]
            
            # Get do_nothing baseline
            baseline = bench_df[bench_df['config'] == 'do_nothing']
            if baseline.empty:
                print(f"\nBenchmark: {benchmark} - No baseline found!")
                continue
            
            baseline_acc = baseline['accuracy'].iloc[0]
            baseline_avg_tpot = baseline['avg_tpot_ms'].iloc[0]
            baseline_throughput = baseline['gen_throughput'].iloc[0]
            
            print(f"\n📊 Benchmark: {benchmark.upper()}")
            print(f"{'Config':<20} {'Accuracy':<12} {'Acc Drop':<12} {'Avg TPOT (ms)':<15} {'Speedup':<10} {'Throughput':<15}")
            print(f"{'-'*95}")
            
            # Sort by config name for consistent ordering
            bench_df = bench_df.sort_values('config')
            
            for _, row in bench_df.iterrows():
                acc_drop = baseline_acc - row['accuracy']
                acc_drop_pct = (acc_drop / baseline_acc * 100) if baseline_acc > 0 else 0
                avg_tpot_speedup = baseline_avg_tpot / row['avg_tpot_ms'] if row['avg_tpot_ms'] > 0 else 0
                
                config_label = row['config']
                if config_label == 'do_nothing':
                    config_label += ' (BASE)'
                
                print(f"{config_label:<20} {row['accuracy']:.4f}       "
                      f"{acc_drop:+.4f} ({acc_drop_pct:+.1f}%)  "
                      f"{row['avg_tpot_ms']:.2f}          "
                      f"{avg_tpot_speedup:.2f}x       "
                      f"{row['gen_throughput']:.1f} tok/s")
    
    # Save detailed results to CSV
    output_file = 'all_benchmark_results.csv'
    df = df.sort_values(['model', 'benchmark', 'config'])
    df.to_csv(output_file, index=False)
    print(f"\n\n💾 Detailed results saved to: {output_file}")
    
    # Create summary statistics by config
    print(f"\n{'='*100}")
    print("SUMMARY: AVERAGE PERFORMANCE ACROSS ALL BENCHMARKS")
    print(f"{'='*100}")
    
    config_summary = defaultdict(lambda: {'speedups': [], 'acc_drops': [], 'models': set(), 'benchmarks': set()})
    
    for model in df['model'].unique():
        for benchmark in df['benchmark'].unique():
            for config in df['config'].unique():
                if config == 'do_nothing':
                    continue
                
                config_df = df[(df['model'] == model) & (df['benchmark'] == benchmark) & (df['config'] == config)]
                baseline = df[(df['model'] == model) & (df['benchmark'] == benchmark) & (df['config'] == 'do_nothing')]
                
                if not config_df.empty and not baseline.empty:
                    speedup = baseline['avg_tpot_ms'].iloc[0] / config_df['avg_tpot_ms'].iloc[0]
                    acc_drop = baseline['accuracy'].iloc[0] - config_df['accuracy'].iloc[0]
                    acc_drop_pct = (acc_drop / baseline['accuracy'].iloc[0] * 100) if baseline['accuracy'].iloc[0] > 0 else 0
                    
                    config_summary[config]['speedups'].append(speedup)
                    config_summary[config]['acc_drops'].append(acc_drop_pct)
                    config_summary[config]['models'].add(model)
                    config_summary[config]['benchmarks'].add(benchmark)
    
    print(f"\n{'Config':<20} {'Avg Speedup':<15} {'Avg Acc Drop %':<20} {'Models':<10} {'Benchmarks':<10}")
    print(f"{'-'*85}")
    
    for config in sorted(config_summary.keys()):
        data = config_summary[config]
        if data['speedups']:
            avg_speedup = sum(data['speedups']) / len(data['speedups'])
            avg_acc_drop = sum(data['acc_drops']) / len(data['acc_drops'])
            num_models = len(data['models'])
            num_benchmarks = len(data['benchmarks'])
            
            print(f"{config:<20} {avg_speedup:.3f}x          "
                  f"{avg_acc_drop:.1f}%              "
                  f"{num_models}          {num_benchmarks}")
    
    # Show best configs for different trade-offs
    print(f"\n{'='*100}")
    print("RECOMMENDATIONS")
    print(f"{'='*100}")
    
    print("\n🚀 Best Speedup (with accuracy drop < 10%):")
    for config in sorted(config_summary.keys()):
        data = config_summary[config]
        if data['speedups'] and data['acc_drops']:
            avg_speedup = sum(data['speedups']) / len(data['speedups'])
            avg_acc_drop = sum(data['acc_drops']) / len(data['acc_drops'])
            if avg_acc_drop < 10:
                print(f"  - {config}: {avg_speedup:.3f}x speedup, {avg_acc_drop:.1f}% accuracy drop")
    
    print("\n🎯 Best Accuracy Preservation (with speedup > 1.05x):")
    for config in sorted(config_summary.keys()):
        data = config_summary[config]
        if data['speedups'] and data['acc_drops']:
            avg_speedup = sum(data['speedups']) / len(data['speedups'])
            avg_acc_drop = sum(data['acc_drops']) / len(data['acc_drops'])
            if avg_speedup > 1.05:
                print(f"  - {config}: {avg_acc_drop:.1f}% accuracy drop, {avg_speedup:.3f}x speedup")

if __name__ == "__main__":
    print("🔍 Collecting all benchmark data (n500 for most, n164 for humaneval)...")
    results = collect_all_benchmark_data()
    
    if results:
        print(f"\n✅ Found {len(results)} result files")
        create_comprehensive_comparison(results)
    else:
        print("❌ No results found!")