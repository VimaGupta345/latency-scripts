#!/usr/bin/env python3
"""Comprehensive visualization of all latency metrics from .metrics files."""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_histogram(filepath, metric_prefix):
    """Extract histogram buckets, count, and sum."""
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


def cumulative_to_per_bucket(buckets):
    edges = []
    counts = []
    prev = 0
    for le, cum in buckets:
        if le == float('inf'):
            break
        edges.append(le)
        counts.append(cum - prev)
        prev = cum
    return edges, counts


def percentile_from_buckets(buckets, count, p):
    target = count * p
    prev_le = 0.0
    prev_count = 0.0
    for le, cum_count in buckets:
        if le == float('inf'):
            return prev_le
        if cum_count >= target:
            bucket_count = cum_count - prev_count
            if bucket_count == 0:
                return le
            fraction = (target - prev_count) / bucket_count
            return prev_le + fraction * (le - prev_le)
        prev_le = le
        prev_count = cum_count
    return prev_le


def plot_histogram_panel(ax, b_edges, b_counts, p_edges, p_counts, title,
                         xlabel, max_val, use_ms=True, show_percentiles=True,
                         b_buckets=None, p_buckets=None, b_count=None, p_count=None):
    """Plot a single histogram comparison panel."""
    scale = 1000 if use_ms else 1
    unit = "ms" if use_ms else "s"

    # Filter to max_val
    b_mask = [i for i, e in enumerate(b_edges) if e * scale <= max_val]
    p_mask = [i for i, e in enumerate(p_edges) if e * scale <= max_val]

    b_e = [b_edges[i] * scale for i in b_mask]
    b_c = [b_counts[i] for i in b_mask]
    p_e = [p_edges[i] * scale for i in p_mask]
    p_c = [p_counts[i] for i in p_mask]

    # Normalize to percentage
    b_total = sum(b_counts)
    p_total = sum(p_counts)
    if b_total == 0 or p_total == 0:
        return
    b_pct = [c / b_total * 100 for c in b_c]
    p_pct = [c / p_total * 100 for c in p_c]

    # Create labels
    labels = []
    prev = 0
    for e in b_e:
        labels.append(f"{prev:.0f}-{e:.0f}" if use_ms else f"{prev:.1f}-{e:.1f}")
        prev = e

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, b_pct, width, label=f'Baseline ({b_total:.0f} obs)',
           color='#4C72B0', alpha=0.85)
    ax.bar(x + width / 2, p_pct, width, label=f'Prowl ({p_total:.0f} obs)',
           color='#DD8452', alpha=0.85)

    ax.set_xlabel(f'{xlabel} ({unit})', fontsize=10)
    ax.set_ylabel('% of observations', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for i, (bp, pp) in enumerate(zip(b_pct, p_pct)):
        if bp > 2:
            ax.text(i - width / 2, bp + 0.5, f'{bp:.1f}%', ha='center',
                    va='bottom', fontsize=6, color='#4C72B0')
        if pp > 2:
            ax.text(i + width / 2, pp + 0.5, f'{pp:.1f}%', ha='center',
                    va='bottom', fontsize=6, color='#DD8452')

    # Add percentile lines
    if show_percentiles and b_buckets and p_buckets and b_count and p_count:
        for pct, ls, lw in [(0.5, '-', 2), (0.9, '--', 1.5), (0.99, ':', 1)]:
            b_val = percentile_from_buckets(b_buckets, b_count, pct) * scale
            p_val = percentile_from_buckets(p_buckets, p_count, pct) * scale
            label_b = f'B P{int(pct*100)}={b_val:.1f}{unit}'
            label_p = f'P P{int(pct*100)}={p_val:.1f}{unit}'
            ax.axvline(x=np.interp(b_val, [0] + b_e, np.arange(-0.5, len(b_e) + 0.5)) - width / 2,
                       color='#4C72B0', linestyle=ls, linewidth=lw, alpha=0.7)
            ax.axvline(x=np.interp(p_val, [0] + p_e, np.arange(-0.5, len(p_e) + 0.5)) + width / 2,
                       color='#DD8452', linestyle=ls, linewidth=lw, alpha=0.7)


def plot_cdf_panel(ax, buckets_b, buckets_p, count_b, count_p, title, xlabel, max_val, use_ms=True):
    """Plot CDF curves."""
    scale = 1000 if use_ms else 1
    unit = "ms" if use_ms else "s"

    # Build CDF points
    def cdf_points(buckets, count):
        xs = [0]
        ys = [0]
        for le, cum in buckets:
            if le == float('inf'):
                break
            if le * scale <= max_val:
                xs.append(le * scale)
                ys.append(cum / count * 100)
        return xs, ys

    bx, by = cdf_points(buckets_b, count_b)
    px, py = cdf_points(buckets_p, count_p)

    ax.plot(bx, by, 'o-', color='#4C72B0', linewidth=2, markersize=4,
            label='Baseline', alpha=0.85)
    ax.plot(px, py, 's-', color='#DD8452', linewidth=2, markersize=4,
            label='Prowl', alpha=0.85)

    # Add P50, P90, P99 horizontal lines
    for pct, ls in [(50, '-'), (90, '--'), (99, ':')]:
        ax.axhline(y=pct, color='gray', linestyle=ls, linewidth=0.8, alpha=0.5)
        ax.text(max_val * 0.98, pct + 1, f'P{pct}', ha='right', fontsize=8, color='gray')

    # Mark percentile values
    for pct_frac, color, marker, label_prefix in [
        (0.5, '#4C72B0', 'o', 'B'), (0.5, '#DD8452', 's', 'P'),
        (0.9, '#4C72B0', 'o', 'B'), (0.9, '#DD8452', 's', 'P'),
    ]:
        if label_prefix == 'B':
            val = percentile_from_buckets(buckets_b, count_b, pct_frac) * scale
        else:
            val = percentile_from_buckets(buckets_p, count_p, pct_frac) * scale
        pct_label = int(pct_frac * 100)
        ax.plot(val, pct_frac * 100, marker, color=color, markersize=8, zorder=5)
        ax.annotate(f'{val:.1f}{unit}', (val, pct_frac * 100),
                    textcoords="offset points", xytext=(5, -12 if label_prefix == 'P' else 5),
                    fontsize=7, color=color, fontweight='bold')

    ax.set_xlabel(f'{xlabel} ({unit})', fontsize=10)
    ax.set_ylabel('CDF (%)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-2, 105)
    ax.set_xlim(0, max_val)


def main():
    baseline_file = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/vgupta345/prowl_related_data/prowl-open-source/results/DeepSeek-Coder-V2-Instruct/humaneval/adv_fp16_deepseek_v2_humaneval_port8020_n164_conf_config_do_nothing_20260316-183713.metrics"
    prowl_file = sys.argv[2] if len(sys.argv) > 2 else \
        "/data/vgupta345/prowl_related_data/prowl-open-source/results/DeepSeek-Coder-V2-Instruct/humaneval/adv_fp16_deepseek_v2_humaneval_port8020_n164_conf_quant_alpha1.175_beta2_optimized_20260316-185640.metrics"
    output_path = sys.argv[3] if len(sys.argv) > 3 else \
        "/data/vgupta345/prowl_related_data/prowl-open-source/results/deepseek_humaneval_all_metrics.png"

    # Parse all histograms
    metrics_config = [
        ("vllm:time_per_output_token_seconds", "TPOT (per token, decode only)", "Time per output token", 150, True),
        ("vllm:e2e_request_latency_seconds", "E2E Request Latency", "E2E latency", 15, False),
        ("vllm:request_decode_time_seconds", "Decode Time (per request)", "Decode time", 15, False),
        ("vllm:time_to_first_token_seconds", "Time to First Token", "TTFT", 3000, True),
    ]

    fig, axes = plt.subplots(len(metrics_config), 2, figsize=(18, 5 * len(metrics_config)))

    for row, (prefix, title, xlabel, max_val, use_ms) in enumerate(metrics_config):
        b_buckets, b_count, b_sum = parse_histogram(baseline_file, prefix)
        p_buckets, p_count, p_sum = parse_histogram(prowl_file, prefix)

        b_edges, b_counts = cumulative_to_per_bucket(b_buckets)
        p_edges, p_counts = cumulative_to_per_bucket(p_buckets)

        # Left: histogram
        plot_histogram_panel(axes[row, 0], b_edges, b_counts, p_edges, p_counts,
                             f'{title} — Distribution', xlabel, max_val, use_ms,
                             b_buckets=b_buckets, p_buckets=p_buckets,
                             b_count=b_count, p_count=p_count)

        # Right: CDF
        plot_cdf_panel(axes[row, 1], b_buckets, p_buckets, b_count, p_count,
                       f'{title} — CDF', xlabel, max_val, use_ms)

        # Add summary stats as text
        if b_count and p_count and b_sum and p_sum:
            scale = 1000 if use_ms else 1
            unit = "ms" if use_ms else "s"
            b_mean = b_sum / b_count * scale
            p_mean = p_sum / p_count * scale
            b_p50 = percentile_from_buckets(b_buckets, b_count, 0.5) * scale
            p_p50 = percentile_from_buckets(p_buckets, p_count, 0.5) * scale
            b_p90 = percentile_from_buckets(b_buckets, b_count, 0.9) * scale
            p_p90 = percentile_from_buckets(p_buckets, p_count, 0.9) * scale
            b_p99 = percentile_from_buckets(b_buckets, b_count, 0.99) * scale
            p_p99 = percentile_from_buckets(p_buckets, p_count, 0.99) * scale

            stats_text = (
                f"Mean: B={b_mean:.1f}{unit} P={p_mean:.1f}{unit} ({b_mean/p_mean:.3f}x)\n"
                f"P50:  B={b_p50:.1f}{unit} P={p_p50:.1f}{unit} ({b_p50/p_p50:.3f}x)\n"
                f"P90:  B={b_p90:.1f}{unit} P={p_p90:.1f}{unit} ({b_p90/p_p90:.3f}x)\n"
                f"P99:  B={b_p99:.1f}{unit} P={p_p99:.1f}{unit} ({b_p99/p_p99:.3f}x)"
            )
            axes[row, 1].text(0.98, 0.35, stats_text, transform=axes[row, 1].transAxes,
                              fontsize=8, verticalalignment='top', horizontalalignment='right',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                              fontfamily='monospace')

    fig.suptitle('DeepSeek-Coder-V2-Instruct — HumanEval\n'
                 'Baseline (do-nothing) vs Prowl (α1.175 β2)  |  bs>1 filter active  |  Seeded (seed=42)',
                 fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
