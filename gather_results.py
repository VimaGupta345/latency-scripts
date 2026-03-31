#!/usr/bin/env python3
"""
Gather accuracy and throughput results from benchmark output files.

Usage:
    python gather_results.py [results_dir]

Walks the results directory tree and extracts:
  - Accuracy from results_*.json files (inside .jsonl directories)
  - Throughput from .metrics files (count/sum of time_per_output_token_seconds)

Prints a summary table to stdout and optionally saves to CSV.
"""

import json
import os
import re
import sys
import glob
import csv
from collections import defaultdict

# Accuracy metric markers per benchmark (prefix-matched against keys in results JSON)
ACCURACY_MARKERS = {
    "humaneval":              "pass@1,",
    "mbpp":                   "pass_at_1,",
    "minerva_math_algebra":   "math_verify,",
    "gsm8k":                  "exact_match,flexible",
    "truthfulqa":             "rougeL_acc,n",
    "truthfulqa_mc2":         "acc,n",
    "squad_completion":       "contains,n",
    "longbench_narrativeqa":  "qa_f1_score,n",
    "coqa":                   "em,n",
    "cnn_dailymail":          "rouge,n",
    "xsum":                   "rouge,n",
    "hotpotqa":               "qa_f1_score,n",
    "triviaqa":               "qa_f1_score,n",
    "squadv2":                "exact,n",
}


def find_accuracy(results_dir):
    """Find all results_*.json files and extract accuracy metrics."""
    accuracies = {}
    
    # Walk looking for results_*.json inside .jsonl directories
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.startswith("results_") and f.endswith(".json"):
                filepath = os.path.join(root, f)
                try:
                    with open(filepath) as fh:
                        data = json.load(fh)
                except (json.JSONDecodeError, IOError):
                    continue
                
                if "results" not in data:
                    continue
                
                # Determine model name and config from path
                # Path structure: results/<model>/<benchmark>/<statfile>.jsonl/<model_sanitized>/results_*.json
                rel = os.path.relpath(filepath, results_dir)
                parts = rel.split(os.sep)
                if len(parts) >= 3:
                    model_name = parts[0]
                    benchmark_dir = parts[1]
                else:
                    continue
                
                # Extract config name from the .jsonl directory name
                jsonl_dir = parts[2] if len(parts) > 2 else ""
                config_match = re.search(r'_conf_(.+)\.jsonl$', jsonl_dir)
                config_name = config_match.group(1) if config_match else "unknown"
                
                # Extract accuracy for each benchmark in results
                for bmk_name, bmk_results in data["results"].items():
                    # Find the matching accuracy marker
                    marker = ACCURACY_MARKERS.get(bmk_name)
                    if marker is None:
                        # Try matching benchmark_dir instead
                        marker = ACCURACY_MARKERS.get(benchmark_dir)
                    
                    accuracy_val = None
                    accuracy_key = None
                    
                    if marker:
                        # Find keys that start with the marker prefix
                        for key, val in bmk_results.items():
                            if key.startswith(marker) and isinstance(val, (int, float)):
                                accuracy_val = val
                                accuracy_key = key
                                break
                    
                    # Fallback: grab first numeric metric that's not stderr/alias
                    if accuracy_val is None:
                        for key, val in bmk_results.items():
                            if isinstance(val, (int, float)) and "stderr" not in key and key != "alias":
                                accuracy_val = val
                                accuracy_key = key
                                break
                    
                    if accuracy_val is not None:
                        key = (model_name, benchmark_dir, config_name)
                        accuracies[key] = {
                            "accuracy": accuracy_val,
                            "metric": accuracy_key,
                            "file": filepath,
                        }
    
    return accuracies


def find_throughput(results_dir):
    """Find all .metrics files and extract throughput (count/sum)."""
    throughputs = {}
    
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".metrics"):
                filepath = os.path.join(root, f)
                rel = os.path.relpath(filepath, results_dir)
                parts = rel.split(os.sep)
                
                if len(parts) >= 2:
                    model_name = parts[0]
                    benchmark_dir = parts[1]
                else:
                    continue
                
                # Extract config name from filename
                config_match = re.search(r'_conf_(.+?)_\d{8}-\d{6}\.metrics$', f)
                config_name = config_match.group(1) if config_match else "unknown"
                
                # Parse the metrics file for count and sum
                count = None
                total = None
                
                try:
                    with open(filepath) as fh:
                        for line in fh:
                            line = line.strip()
                            if line.startswith("vllm:time_per_output_token_seconds_count{engine=\"0\""):
                                count = float(line.split()[-1])
                            elif line.startswith("vllm:time_per_output_token_seconds_sum{engine=\"0\""):
                                total = float(line.split()[-1])
                except IOError:
                    continue
                
                if count and total and total > 0:
                    throughput = count / total  # tokens per second
                    key = (model_name, benchmark_dir, config_name)
                    throughputs[key] = {
                        "throughput_tps": throughput,
                        "count": count,
                        "sum": total,
                        "file": filepath,
                    }
    
    return throughputs


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    
    print(f"Scanning: {results_dir}")
    print()
    
    accuracies = find_accuracy(results_dir)
    throughputs = find_throughput(results_dir)
    
    # Merge all keys
    all_keys = sorted(set(list(accuracies.keys()) + list(throughputs.keys())))
    
    if not all_keys:
        print("No results found.")
        return
    
    # Print table
    header = f"{'Model':<40} {'Benchmark':<25} {'Config':<40} {'Accuracy':<20} {'Metric':<30} {'Throughput (tok/s)':<20}"
    print(header)
    print("=" * len(header))
    
    rows = []
    for key in all_keys:
        model, benchmark, config = key
        
        acc_info = accuracies.get(key, {})
        thr_info = throughputs.get(key, {})
        
        accuracy = acc_info.get("accuracy")
        metric = acc_info.get("metric", "")
        throughput = thr_info.get("throughput_tps")
        
        acc_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"
        thr_str = f"{throughput:.2f}" if throughput is not None else "N/A"
        
        print(f"{model:<40} {benchmark:<25} {config:<40} {acc_str:<20} {metric:<30} {thr_str:<20}")
        rows.append([model, benchmark, config, acc_str, metric, thr_str])
    
    # Save CSV
    csv_path = os.path.join(results_dir, "summary.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Model", "Benchmark", "Config", "Accuracy", "Metric", "Throughput (tok/s)"])
        writer.writerows(rows)
    
    print()
    print(f"Summary saved to: {csv_path}")


if __name__ == "__main__":
    main()
