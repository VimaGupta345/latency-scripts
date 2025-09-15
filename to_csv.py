#!/usr/bin/env python3
"""
metrics_to_csv.py

Parses blocks like either:

==> <path>.metrics
modified: <timestamp>
  Total Prompt Tokens: ... Total Generated Tokens: ... Avg Prompt Tokens/Req: ... Avg Generated Tokens/Req: ... Average Input/Output Ratio: ... Total Requests: ...

or:

==> <path>.metrics
modified: <timestamp>
  TPOT: 14.97 [ms]

…and emits a CSV with unioned columns.

Usage examples:
  # Full token metrics (e.g., humaneval)
  your_command | python metrics_to_csv.py > out.csv

  # TPOT-only metrics (e.g., gsm8k)
  your_other_command | python metrics_to_csv.py > out.csv

  # From a file:
  python metrics_to_csv.py -i sample.txt -o out.csv
"""

import re
import os
import sys
import csv
import argparse
from typing import Optional, Dict, Any

METRICS_RE = re.compile(
    r"Total Prompt Tokens:\s*(?P<tpt>\d+)\s+"
    r"Total Generated Tokens:\s*(?P<tgt>\d+)\s+"
    r"Avg Prompt Tokens/Req:\s*(?P<aptr>[\d.]+)\s+"
    r"Avg Generated Tokens/Req:\s*(?P<agtr>[\d.]+)\s+"
    r"Average Input/Output Ratio:\s*(?P<ratio>[\d.]+)\s+"
    r"Total Requests:\s*(?P<trq>\d+)"
)

TPOT_RE = re.compile(r"\bTPOT:\s*(?P<tpot>[\d.]+)\s*\[ms\]", re.IGNORECASE)

# Optional fields we try to extract from the file path/name
DATE_TIME_IN_NAME_RE = re.compile(r"_(?P<date>\d{8})-(?P<time>\d{6})\.metrics$")
PORT_RE  = re.compile(r"port(?P<port>\d+)")
BATCH_RE = re.compile(r"batch(?P<batch>\d+)")
N_RE     = re.compile(r"\bn(?P<n>\d+)\b")
ALPHA_RE = re.compile(r"alpha(?P<alpha>\d+)")
BETA_RE  = re.compile(r"beta(?P<beta>\d+)")
CONF_RE  = re.compile(r"\bconf_(?P<conf>[^_]+)\b")
OPTIMIZED_RE = re.compile(r"\boptimized\b")

def parse_from_path(full_path: str) -> Dict[str, Any]:
    """Best-effort extraction of useful bits from the path."""
    out: Dict[str, Any] = {}
    norm = os.path.normpath(full_path)
    parts = norm.split(os.sep)

    # Expect .../quality/<dataset>/<eval>/<model>/<filename>
    try:
        q_idx = parts.index('quality')
        out['dataset'] = parts[q_idx + 1] if len(parts) > q_idx + 1 else ''
        out['eval_kind'] = parts[q_idx + 2] if len(parts) > q_idx + 2 else ''
        out['model'] = parts[q_idx + 3] if len(parts) > q_idx + 3 else ''
    except ValueError:
        out['dataset'] = ''
        out['eval_kind'] = ''
        out['model'] = ''

    base = os.path.basename(full_path)
    base_no_ext = base.rsplit('.metrics', 1)[0]

    # Default batch to 16 unless overridden by filename
    out['batch'] = '16'

    # Run/date embedded in filename
    m = DATE_TIME_IN_NAME_RE.search(base)
    if m:
        out['run_date'] = m.group('date')   # e.g., 20250914
        out['run_time'] = m.group('time')   # e.g., 161708
    else:
        out['run_date'] = ''
        out['run_time'] = ''

    # Misc fields (optional)
    def grab(rx, key):
        mm = rx.search(base_no_ext)
        if mm:
            out[key] = mm.group(key)
    grab(PORT_RE, 'port')
    grab(BATCH_RE, 'batch')   # overrides default if present
    grab(N_RE, 'n')
    grab(ALPHA_RE, 'alpha')
    grab(BETA_RE, 'beta')

    mconf = CONF_RE.search(base_no_ext)
    out['conf'] = mconf.group('conf') if mconf else ''

    out['optimized'] = 'yes' if OPTIMIZED_RE.search(base_no_ext) else 'no'
    out['file_name'] = base
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', help='Input text file (defaults to stdin)')
    ap.add_argument('-o', '--output', help='Output CSV file (defaults to stdout)')
    args = ap.parse_args()

    if args.input:
        fin = open(args.input, 'r', encoding='utf-8')
    else:
        fin = sys.stdin

    rows = []
    current: Optional[Dict[str, Any]] = None

    def flush_current():
        if current is None:
            return
        # Write only if we captured any metrics line
        if ('total_prompt_tokens' in current) or ('tpot_ms' in current):
            rows.append(current.copy())

    for raw in fin:
        line = raw.rstrip('\n')

        if line.startswith('==> '):
            # New block
            flush_current()
            path = line[4:].strip()
            current = {'path': path}
            current.update(parse_from_path(path))

        elif line.startswith('modified: '):
            if current is not None:
                current['modified'] = line.split('modified:', 1)[1].strip()

        elif 'Total Prompt Tokens:' in line:
            m = METRICS_RE.search(line)
            if m and current is not None:
                current['total_prompt_tokens'] = int(m.group('tpt'))
                current['total_generated_tokens'] = int(m.group('tgt'))
                current['avg_prompt_tokens_per_req'] = float(m.group('aptr'))
                current['avg_generated_tokens_per_req'] = float(m.group('agtr'))
                current['avg_io_ratio'] = float(m.group('ratio'))
                current['total_requests'] = int(m.group('trq'))

        elif 'TPOT:' in line:
            m = TPOT_RE.search(line)
            if m and current is not None:
                current['tpot_ms'] = float(m.group('tpot'))

        else:
            # ignore separators like "--" and blank lines
            pass

    # Final one
    flush_current()

    # Column order (union of all known fields)
    fieldnames = [
        'dataset', 'eval_kind', 'model',
        'port', 'batch', 'n', 'alpha', 'beta', 'conf', 'optimized',
        'run_date', 'run_time',
        'modified',
        'tpot_ms',  # present for TPOT runs
        'total_prompt_tokens', 'total_generated_tokens',
        'avg_prompt_tokens_per_req', 'avg_generated_tokens_per_req',
        'avg_io_ratio', 'total_requests',
        'file_name', 'path'
    ]

    # Write CSV
    if args.output:
        fout = open(args.output, 'w', newline='', encoding='utf-8')
    else:
        fout = sys.stdout

    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, '') for k in fieldnames})

    if args.input:
        fin.close()
    if args.output and fout is not sys.stdout:
        fout.close()

if __name__ == '__main__':
    main()

