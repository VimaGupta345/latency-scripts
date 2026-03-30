#!/usr/bin/env python3
"""
Post-process all_results.csv for easier viewing.

Filters out:
  - Benchmarks: sharegpt, squad_completion, truthfulqa, xsum, hotpotqa,
    longbench_narrativeqa, coqa, cnn_dailymail
  - Rows that ran in eager mode (cuda_graph_enabled=False)
  - Rows with only coarse histogram TPOT (no direct or fine histogram)
  - Rows without any accuracy value

Removes columns that are entirely empty after filtering.

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


# Columns to always drop from the filtered output
FORCE_DROP_COLS = {"hist_n_buckets", "hist_granularity", "source"}

EXCLUDE_BENCHMARKS = {
    "sharegpt",
    "squad_completion",
    "sqaud_completion",
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
            row["tp"], row["batch_size"], row["source"],
            row.get("accuracy_metric", ""))


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
    n_bmk = n_eager = n_coarse = n_no_acc = n_no_tpot = 0
    filtered = []
    for row in rows:
        if row["benchmark"].lower() in EXCLUDE_BENCHMARKS:
            n_bmk += 1
            continue
        if row.get("cuda_graph_enabled", "").strip() == "False":
            n_eager += 1
            continue
        # Check TPOT availability
        has_direct = bool(row.get("p50_tpot_ms_direct", "").strip())
        has_fine = bool(row.get("p50_tpot_ms_hist_fine", "").strip())
        has_coarse = bool(row.get("p50_tpot_ms_hist_coarse", "").strip())
        # Remove rows whose only TPOT comes from a coarse histogram
        if has_coarse and not has_direct and not has_fine:
            n_coarse += 1
            continue
        # Remove rows without any TPOT data
        if not has_direct and not has_fine and not has_coarse:
            n_no_tpot += 1
            continue
        # Remove rows without any accuracy
        if not row.get("accuracy", "").strip():
            n_no_acc += 1
            continue
        filtered.append(row)

    # Convert accuracy to percentage (×100); delta is already ×100 from gather script
    for row in filtered:
        acc = row.get("accuracy", "").strip()
        if acc:
            try:
                row["accuracy"] = f"{float(acc) * 100:.2f}"
            except ValueError:
                pass

    # Drop columns that are entirely empty or force-dropped
    non_empty_cols = set()
    for row in filtered:
        for col in fieldnames:
            if row.get(col, "").strip():
                non_empty_cols.add(col)
    kept_cols = [c for c in fieldnames
                 if c in non_empty_cols and c not in FORCE_DROP_COLS]
    dropped_cols = [c for c in fieldnames if c not in kept_cols]

    # Sort: within each model, colocated groups first, then disagg
    MODE_ORDER = {"colocated": 0, "disagg": 1}
    filtered.sort(key=lambda r: (
        r["model"],
        MODE_ORDER.get(r["mode"], 2),
        r["benchmark"],
        r["batch_size"],
        r["source"],
        r["is_baseline"] != "True",  # baselines first
        r["config"],
    ))

    # Insert empty rows between groups
    output_rows = []
    prev_key = None
    for row in filtered:
        cur_key = group_key(row)
        if prev_key is not None and cur_key != prev_key:
            output_rows.append({col: "" for col in kept_cols})
        output_rows.append(row)
        prev_key = cur_key

    # Write
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=kept_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Input:    {len(rows)} rows")
    print(f"Filtered: {len(filtered)} rows ({len(rows) - len(filtered)} removed)")
    print(f"  Benchmark exclusions:  {n_bmk}")
    print(f"  Eager mode removals:   {n_eager}")
    print(f"  Coarse-only removals:  {n_coarse}")
    print(f"  No TPOT removals:      {n_no_tpot}")
    print(f"  No accuracy removals:  {n_no_acc}")
    n_groups = len(set(group_key(r) for r in filtered))
    print(f"Groups:   {n_groups}")
    print(f"Columns:  {len(kept_cols)} kept, {len(dropped_cols)} dropped")
    if dropped_cols:
        print(f"  Dropped: {', '.join(dropped_cols)}")
    print(f"Output:   {output_path}")


if __name__ == "__main__":
    main()
