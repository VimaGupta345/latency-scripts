#!/usr/bin/env python3
"""
Post-process all_results.csv for easier viewing.

Filters out:
  - Benchmarks: sharegpt, squad_completion, truthfulqa, xsum, hotpotqa,
    longbench_narrativeqa, coqa, cnn_dailymail
  - Rows that ran in eager mode (cuda_graph_enabled=False)

Inserts empty separator rows between baseline groups so it's easy to see
which prowl configs are compared against which baseline.

Usage:
    # Default: reads results/all_results.csv, writes results/all_results_filtered.csv
    python latency-scripts/process_results.py

    # Custom input/output
    python latency-scripts/process_results.py results/all_results.csv -o filtered.csv
"""

import argparse
import csv
import os


EXCLUDE_BENCHMARKS = {
    "sharegpt",
    "squad_completion",
    "truthfulqa",
    "xsum",
    "hotpotqa",
    "longbench_narrativeqa",
    "coqa",
    "cnn_dailymail",
}


def group_key(row):
    """Key that defines a baseline group (baseline + its prowl variants)."""
    return (row["model"], row["benchmark"], row["mode"],
            row["batch_size"], row["source"])


def main():
    parser = argparse.ArgumentParser(description="Filter and format all_results.csv.")
    parser.add_argument("input", nargs="?",
                        default="/data/vgupta345/prowl_related_data/prowl-open-source/results/all_results.csv",
                        help="Input CSV (default: results/all_results.csv)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV (default: {input_dir}/all_results_filtered.csv)")
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        os.path.dirname(args.input), "all_results_filtered.csv")

    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Filter
    filtered = []
    for row in rows:
        if row["benchmark"].lower() in EXCLUDE_BENCHMARKS:
            continue
        if row.get("cuda_graph_enabled", "").strip() == "False":
            continue
        filtered.append(row)

    # Insert empty rows between groups
    output_rows = []
    prev_key = None
    for row in filtered:
        cur_key = group_key(row)
        if prev_key is not None and cur_key != prev_key:
            output_rows.append({fn: "" for fn in fieldnames})
        output_rows.append(row)
        prev_key = cur_key

    # Write
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Input:    {len(rows)} rows")
    print(f"Filtered: {len(filtered)} rows ({len(rows) - len(filtered)} removed)")
    print(f"  Benchmark exclusions: {sum(1 for r in rows if r['benchmark'].lower() in EXCLUDE_BENCHMARKS)}")
    print(f"  Eager mode removals:  {sum(1 for r in rows if r.get('cuda_graph_enabled','').strip() == 'False')}")
    n_groups = len(set(group_key(r) for r in filtered))
    print(f"Groups:   {n_groups}")
    print(f"Output:   {output_path}")


if __name__ == "__main__":
    main()
