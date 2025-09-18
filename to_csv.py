#!/usr/bin/env python3
"""
Parse metrics scan output into a CSV.

Usage:
  # Read from stdin, write to stdout
  cat metrics_dump.txt | python metrics_to_csv.py

  # Read from a file, write to a CSV file
  python metrics_to_csv.py metrics_dump.txt -o runs.csv
"""

import sys
import os
import re
import csv
import argparse
from datetime import datetime

# --- Regexes for pieces we care about ---
RE_PATH_LINE = re.compile(r"^==>\s*(.+)$")
RE_MODIFIED  = re.compile(r"^modified:\s*(.+)$")
RE_TPOT      = re.compile(r"TPOT:\s*([0-9.]+)\s*\[ms\]", re.IGNORECASE)

# Things pulled from filename stem
RE_N         = re.compile(r"_n(?P<n>\d+)")
RE_PORT      = re.compile(r"_port(?P<port>\d+)")
RE_CONF      = re.compile(r"_conf_(?P<conf>[^_]+)")
RE_ALPHA     = re.compile(r"_alpha(?P<alpha>[0-9.]+)")
RE_BETA      = re.compile(r"_beta(?P<beta>[0-9.]+)")
RE_OPTIMIZED = re.compile(r"_optimized(?=_|$)")
RE_QUANT     = re.compile(r"_quant(?=_|$)")
RE_RUNID     = re.compile(r"_(?P<runid>\d{8}-\d{6})$")

def parse_runid_iso(runid: str) -> str:
    """Convert run id like 20250915-031642 to ISO 'YYYY-MM-DD HH:MM:SS'."""
    try:
        dt = datetime.strptime(runid, "%Y%m%d-%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""

def parse_block(block_lines):
    """
    Parse one block of lines (path, modified, optional TPOT).
    Returns a dict of fields.
    """
    rec = {}
    path = None
    modified = None
    tpot = None

    for ln in block_lines:
        ln = ln.rstrip("\n")
        m = RE_PATH_LINE.match(ln)
        if m:
            path = m.group(1).strip()
            continue
        m = RE_MODIFIED.match(ln)
        if m:
            modified = m.group(1).strip()
            continue
        m = RE_TPOT.search(ln)
        if m:
            tpot = m.group(1).strip()
            continue

    if not path:
        return None  # not a valid block

    rec["file_path"] = path
    rec["modified_time"] = modified or ""
    rec["tpot_ms"] = tpot or ""

    # Break the path up
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)

    # Try to infer dataset, eval_type (e.g., ngram), model_name from path structure:
    # ../stats/quality/<dataset>/<eval_type>/<ModelName>/<filename>.metrics
    try:
        q_idx = parts.index("quality")
        rec["dataset"]   = parts[q_idx + 1] if len(parts) > q_idx + 1 else ""
        rec["eval_type"] = parts[q_idx + 2] if len(parts) > q_idx + 2 else ""
        rec["model_name"] = parts[q_idx + 3] if len(parts) > q_idx + 3 else ""
    except ValueError:
        # 'quality' not found; fall back to best-effort
        rec["dataset"] = rec.get("dataset","")
        rec["eval_type"] = rec.get("eval_type","")
        rec["model_name"] = rec.get("model_name","")

    filename = parts[-1] if parts else ""
    rec["file_name"] = filename
    stem = filename[:-8] if filename.endswith(".metrics") else os.path.splitext(filename)[0]
    rec["file_stem"] = stem

    # Pull standard tokens from the stem
    # n (examples)
    m = RE_N.search(stem)
    rec["n_examples"] = m.group("n") if m else ""

    # port
    m = RE_PORT.search(stem)
    rec["port"] = m.group("port") if m else ""

    # config name (the token immediately after 'conf_')
    m = RE_CONF.search(stem)
    rec["config_name"] = m.group("conf") if m else ""

    # alpha, beta
    m = RE_ALPHA.search(stem)
    rec["alpha"] = m.group("alpha") if m else ""
    m = RE_BETA.search(stem)
    rec["beta"] = m.group("beta") if m else ""

    # quant/optimized flags
    rec["quant_flag"] = "true" if RE_QUANT.search(stem) else "false"
    rec["optimized_flag"] = "true" if RE_OPTIMIZED.search(stem) else "false"

    # Run-id timestamp embedded at the end of the stem
    m = RE_RUNID.search(stem)
    runid = m.group("runid") if m else ""
    rec["run_id_ts"] = runid
    rec["run_id_iso"] = parse_runid_iso(runid) if runid else ""

    # Try to parse early tokens for exec mode / precision (e.g., adv_fp16_*)
    # This is best-effort and won't break if absent.
    # e.g., 'adv_fp16_deepseek_squadv2_port...'
    tokens = stem.split("_")
    if len(tokens) >= 2:
        rec["exec_mode"] = tokens[0]
        # precision is often like fp16/fp8/etc
        rec["precision"] = tokens[1] if tokens[1].startswith("fp") else ""
    else:
        rec["exec_mode"] = ""
        rec["precision"] = ""

    return rec

def parse_stream(text: str):
    """
    Split the input into logical blocks. A block begins with a line that starts with '==>'
    and continues until a blank line or the start of the next block.
    """
    records = []
    cur = []

    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if RE_PATH_LINE.match(ln):
            # flush previous block
            if cur:
                rec = parse_block(cur)
                if rec:
                    records.append(rec)
                cur = []
            cur.append(ln)
        else:
            # add to current until blank line signifies end
            cur.append(ln)
            if ln.strip() == "":
                rec = parse_block(cur)
                if rec:
                    records.append(rec)
                cur = []

    # flush last
    if cur:
        rec = parse_block(cur)
        if rec:
            records.append(rec)

    return records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="Input file (defaults to stdin)")
    ap.add_argument("-o", "--output", help="Output CSV file (defaults to stdout)")
    args = ap.parse_args()

    # Read input
    if args.input and args.input != "-":
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    records = parse_stream(text)

    if not records:
        # Still write headers so the pipeline doesn't break
        records = [dict()]

    # Priority columns first
    priority = [
        "model_name",
        "n_examples",
        "config_name",
        "alpha",
        "beta",
        "tpot_ms",
        "modified_time",
    ]

    # Gather all keys seen
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())

    # Remove priority ones, then append the rest in a stable order
    rest = [k for k in sorted(all_keys) if k not in priority]

    fieldnames = priority + rest

    # Output
    if args.output:
        outfh = open(args.output, "w", newline="", encoding="utf-8")
        close_out = True
    else:
        outfh = sys.stdout
        close_out = False

    writer = csv.DictWriter(outfh, fieldnames=fieldnames)
    writer.writeheader()
    for r in records:
        writer.writerow(r)

    if close_out:
        outfh.close()

if __name__ == "__main__":
    main()

