#!/usr/bin/env python3
"""
Run multimodal benchmarks (ChartQA, MMMU, etc.) using vllm-vlm offline inference.

Multimodal tasks require image inputs that can't go through the text-only
local-completions API, so we use vllm-vlm which loads the model directly.

Usage (called by run_llama4_multimodal_benchmarks.sh):
    python lm_eval_multimodal_serve.py \
        -m meta-llama/Llama-4-Scout-17B-16E-Instruct \
        -o stat_filename \
        -b chartqa \
        -cf /path/to/config.json \
        -tp 2
"""

import os
import argparse
import subprocess


bmk_defaults = {
    "chartqa": {
        "limit": 2500,
        "tasks": "chartqa",
        "extra_args": "--trust_remote_code",
    },
    "mmmu_val": {
        "limit": 900,
        "tasks": "mmmu_val",
        "extra_args": "--trust_remote_code",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model path or name.")
    parser.add_argument("-o", "--stat_filename", required=True, help="Base stat filename.")
    parser.add_argument("-b", "--benchmark", required=True, help="Benchmark to run (chartqa, mmmu_val).")
    parser.add_argument("-l", "--limit", type=float, default=None, help="Number of samples to limit.")
    parser.add_argument("-cf", "--config_file", required=True, help="Prowl config file.")
    parser.add_argument("-tp", "--tp_size", type=int, default=2, help="Tensor parallel size.")
    parser.add_argument("-mb", "--max_batch_size", type=int, default=16, help="Max batch size for lm-eval.")
    args = parser.parse_args()

    cfg = bmk_defaults.get(args.benchmark)
    if cfg is None:
        raise ValueError(f"Unknown benchmark: {args.benchmark}. Available: {list(bmk_defaults.keys())}")

    limit = int(args.limit) if args.limit else cfg["limit"]

    model_name = os.path.basename(args.model.rstrip("/"))
    conf_name = os.path.basename(args.config_file).replace(".json", "")
    tmp_home = os.environ.get("TMP_HOME", "/data/vgupta345/prowl_related_data/prowl-open-source")
    stats_dir = os.path.join(tmp_home, "results", model_name, args.benchmark)
    os.makedirs(stats_dir, exist_ok=True)

    stat_file = f"{args.stat_filename}_n{limit}_conf_{conf_name}"

    model_args = (
        f"pretrained={args.model},"
        f"tensor_parallel_size={args.tp_size},"
        f"gpu_memory_utilization=0.9,"
        f"max_model_len=4096,"
        f"max_num_seqs={args.max_batch_size},"
        f"mixtral_config_file={args.config_file},"
        f"trust_remote_code=True,"
        f"dtype=auto"
    )

    lm_eval_cmd = (
        f"lm-eval --model vllm-vlm "
        f"--tasks {cfg['tasks']} "
        f"--model_args {model_args} "
        f"--limit {limit} --log_samples --seed 42 --apply_chat_template "
        f"{cfg['extra_args']} "
        f"--output_path {stats_dir}/{stat_file}.jsonl "
        f"2>&1 | tee {stats_dir}/{stat_file}.log"
    )
    print(f"eval_cmd: {lm_eval_cmd}")
    subprocess.run(lm_eval_cmd, shell=True)

    print("Done.")


if __name__ == "__main__":
    main()
