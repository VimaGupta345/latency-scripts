#!/usr/bin/env python3
"""Parse [LYNX ROUTING] lines from vLLM server logs to summarize
expert activation counts before and after PROWL.

Usage:
    python parse_expert_counts.py <server_log_file>
    python parse_expert_counts.py <server_log_file> --csv output.csv

The server log files live under:
    stats/perf/spec_decode/ngram/<model_name>/<stat_file>.log
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

LYNX_PATTERN = re.compile(
    r"\[LYNX ROUTING\].*?"
    r"Batch\s+(?P<batch>\d+)\s+"
    r"layer\s+(?P<layer>\S+)\s+"
    r"policy\s+(?P<policy>\S+)\s+"
    r"activated experts\s+"
    r"before-prowl=(?P<before>\d+)/(?P<total>\d+),\s*"
    r"after-prowl=(?P<after>\d+)/(?P<total2>\d+),\s*"
    r"delta=(?P<delta>[+-]?\d+)"
)

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def parse_log(path: str):
    records = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            cleaned = ANSI_ESCAPE.sub("", line)
            m = LYNX_PATTERN.search(cleaned)
            if m:
                records.append({
                    "batch": int(m.group("batch")),
                    "layer": m.group("layer"),
                    "policy": m.group("policy"),
                    "before": int(m.group("before")),
                    "after": int(m.group("after")),
                    "total": int(m.group("total")),
                    "delta": int(m.group("delta")),
                })
    return records


def summarize(records):
    if not records:
        print("No [LYNX ROUTING] entries found.")
        return

    policy = records[0]["policy"]
    total_experts = records[0]["total"]
    n = len(records)

    befores = [r["before"] for r in records]
    afters = [r["after"] for r in records]
    deltas = [r["delta"] for r in records]

    avg_before = sum(befores) / n
    avg_after = sum(afters) / n
    avg_delta = sum(deltas) / n
    reduction_pct = (1 - avg_after / avg_before) * 100 if avg_before > 0 else 0

    print("=" * 65)
    print("PROWL Expert Activation Summary")
    print("=" * 65)
    print(f"  Policy:            {policy}")
    print(f"  Total experts:     {total_experts}")
    print(f"  Observations:      {n}")
    print()
    print("  BEFORE PROWL (unique experts activated per routing call):")
    print(f"    Mean:  {avg_before:.1f}")
    print(f"    Min:   {min(befores)}")
    print(f"    Max:   {max(befores)}")
    print()
    print("  AFTER PROWL:")
    print(f"    Mean:  {avg_after:.1f}")
    print(f"    Min:   {min(afters)}")
    print(f"    Max:   {max(afters)}")
    print()
    print("  DELTA (after - before):")
    print(f"    Mean:  {avg_delta:+.1f}")
    print(f"    Min:   {min(deltas):+d}")
    print(f"    Max:   {max(deltas):+d}")
    print()
    print(f"  Avg expert reduction: {reduction_pct:.1f}%")
    print(f"  Avg experts kept:     {avg_after:.1f} / {total_experts}")
    print("=" * 65)

    batches = defaultdict(lambda: {"before": [], "after": []})
    for r in records:
        batches[r["batch"]]["before"].append(r["before"])
        batches[r["batch"]]["after"].append(r["after"])

    if len(batches) > 1:
        print(f"\n  Per-batch breakdown ({len(batches)} batches):")
        print(f"  {'Batch':>6}  {'Avg Before':>10}  {'Avg After':>10}  {'Reduction':>10}")
        for bid in sorted(batches.keys()):
            b = batches[bid]
            ab = sum(b["before"]) / len(b["before"])
            aa = sum(b["after"]) / len(b["after"])
            red = (1 - aa / ab) * 100 if ab > 0 else 0
            print(f"  {bid:>6}  {ab:>10.1f}  {aa:>10.1f}  {red:>9.1f}%")


def write_csv(records, csv_path):
    if not records:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\nCSV written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse PROWL expert activation counts from vLLM server logs.")
    parser.add_argument("log_file", help="Path to the vLLM server log file")
    parser.add_argument("--csv", default=None,
                        help="Optional: write raw records to CSV")
    args = parser.parse_args()

    records = parse_log(args.log_file)
    summarize(records)
    if args.csv:
        write_csv(records, args.csv)


if __name__ == "__main__":
    main()
