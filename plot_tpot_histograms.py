#!/usr/bin/env python3
"""Plot TPOT histogram comparison between baseline and prowl from .metrics files."""

import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_histogram_buckets(filepath):
    """Extract TPOT histogram buckets from a Prometheus .metrics file."""
    buckets = []  # list of (le, cumulative_count)
    with open(filepath) as f:
        for line in f:
            if line.startswith("vllm:time_per_output_token_seconds_bucket{"):
                # Extract le value
                le_start = line.index('le="') + 4
                le_end = line.index('"', le_start)
                le_val = line[le_start:le_end]
                count = float(line.split()[-1])
                if le_val == "+Inf":
                    le_val = float('inf')
                else:
                    le_val = float(le_val)
                buckets.append((le_val, count))
    return buckets

def cumulative_to_per_bucket(buckets):
    """Convert cumulative histogram to per-bucket counts."""
    edges = []
    counts = []
    prev_count = 0
    for le, cum_count in buckets:
        if le == float('inf'):
            break
        edges.append(le)
        counts.append(cum_count - prev_count)
        prev_count = cum_count
    return edges, counts

def main():
    baseline_file = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/vgupta345/prowl_related_data/prowl-open-source/results/DeepSeek-Coder-V2-Instruct/humaneval/adv_fp16_deepseek_v2_humaneval_port8020_n164_conf_config_do_nothing_20260316-183713.metrics"
    prowl_file = sys.argv[2] if len(sys.argv) > 2 else \
        "/data/vgupta345/prowl_related_data/prowl-open-source/results/DeepSeek-Coder-V2-Instruct/humaneval/adv_fp16_deepseek_v2_humaneval_port8020_n164_conf_quant_alpha1.175_beta2_optimized_20260316-185640.metrics"
    output_path = sys.argv[3] if len(sys.argv) > 3 else \
        "/data/vgupta345/prowl_related_data/prowl-open-source/results/tpot_histogram_comparison.png"

    baseline_buckets = parse_histogram_buckets(baseline_file)
    prowl_buckets = parse_histogram_buckets(prowl_file)

    b_edges, b_counts = cumulative_to_per_bucket(baseline_buckets)
    p_edges, p_counts = cumulative_to_per_bucket(prowl_buckets)

    # Convert to ms for readability
    b_edges_ms = [e * 1000 for e in b_edges]
    p_edges_ms = [e * 1000 for e in p_edges]

    # Only plot buckets up to 300ms (everything above is 0)
    max_ms = 300
    b_mask = [i for i, e in enumerate(b_edges_ms) if e <= max_ms]
    p_mask = [i for i, e in enumerate(p_edges_ms) if e <= max_ms]

    b_edges_plot = [b_edges_ms[i] for i in b_mask]
    b_counts_plot = [b_counts[i] for i in b_mask]
    p_edges_plot = [p_edges_ms[i] for i in p_mask]
    p_counts_plot = [p_counts[i] for i in p_mask]

    # Normalize to percentages
    b_total = sum(b_counts)
    p_total = sum(p_counts)
    b_pct = [c / b_total * 100 for c in b_counts_plot]
    p_pct = [c / p_total * 100 for c in p_counts_plot]

    # Create labels from bucket edges: "(prev, le]"
    def make_labels(edges_ms):
        labels = []
        prev = 0
        for e in edges_ms:
            labels.append(f"{prev}-{int(e)}")
            prev = int(e)
        return labels

    b_labels = make_labels(b_edges_plot)
    p_labels = make_labels(p_edges_plot)

    # Since both have the same bucket boundaries, use one set of labels
    labels = b_labels

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left panel: absolute counts ---
    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width/2, b_counts_plot, width, label='Baseline (do-nothing)', color='#4C72B0', alpha=0.85)
    ax1.bar(x + width/2, p_counts_plot, width, label='Prowl (α1.175 β2)', color='#DD8452', alpha=0.85)
    ax1.set_xlabel('TPOT bucket (ms)', fontsize=12)
    ax1.set_ylabel('Observation count', fontsize=12)
    ax1.set_title('TPOT Distribution — Absolute Counts', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # Add count annotations on top of bars
    for i, (bc, pc) in enumerate(zip(b_counts_plot, p_counts_plot)):
        if bc > 0:
            ax1.text(i - width/2, bc + 100, f'{bc:,.0f}', ha='center', va='bottom', fontsize=7, color='#4C72B0')
        if pc > 0:
            ax1.text(i + width/2, pc + 100, f'{pc:,.0f}', ha='center', va='bottom', fontsize=7, color='#DD8452')

    # --- Right panel: percentage ---
    ax2.bar(x - width/2, b_pct, width, label='Baseline (do-nothing)', color='#4C72B0', alpha=0.85)
    ax2.bar(x + width/2, p_pct, width, label='Prowl (α1.175 β2)', color='#DD8452', alpha=0.85)
    ax2.set_xlabel('TPOT bucket (ms)', fontsize=12)
    ax2.set_ylabel('% of observations', fontsize=12)
    ax2.set_title('TPOT Distribution — Normalized', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    for i, (bp, pp) in enumerate(zip(b_pct, p_pct)):
        if bp > 0.5:
            ax2.text(i - width/2, bp + 0.5, f'{bp:.1f}%', ha='center', va='bottom', fontsize=7, color='#4C72B0')
        if pp > 0.5:
            ax2.text(i + width/2, pp + 0.5, f'{pp:.1f}%', ha='center', va='bottom', fontsize=7, color='#DD8452')

    fig.suptitle('DeepSeek-Coder-V2-Instruct — HumanEval TPOT (decode only, bs>1 filter active)\n'
                 f'Baseline: {b_total:,.0f} obs, avg {sum(b_counts_plot) and (sum([b_counts[i]*b_edges[i] for i in b_mask])/b_total*1000):.1f}ms  |  '
                 f'Prowl: {p_total:,.0f} obs, avg {sum(p_counts_plot) and (sum([p_counts[i]*p_edges[i] for i in p_mask])/p_total*1000):.1f}ms',
                 fontsize=11, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")

    # Also print summary table
    print(f"\n{'Bucket (ms)':<15} {'Baseline':>10} {'Baseline%':>10} {'Prowl':>10} {'Prowl%':>10}")
    print("-" * 55)
    for i, label in enumerate(labels):
        bc = b_counts_plot[i]
        pc = p_counts_plot[i]
        bp = b_pct[i]
        pp = p_pct[i]
        if bc > 0 or pc > 0:
            print(f"{label:<15} {bc:>10,.0f} {bp:>9.1f}% {pc:>10,.0f} {pp:>9.1f}%")

if __name__ == "__main__":
    main()
