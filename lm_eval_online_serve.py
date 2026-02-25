#!/usr/bin/env python3

import os
import time
import signal
import argparse
import subprocess
import threading
import shlex
import urllib.request
import urllib.error
from datetime import datetime

DEFAULT_QUALITY_STATS_ROOT = "/nethome/rdudala3/latency-scripts/stats/quality"
QUALITY_STATS_ROOT = os.path.expanduser(
    os.environ.get("QUALITY_STATS_ROOT", DEFAULT_QUALITY_STATS_ROOT))

# Default settings per benchmark
bmk_defaults = {
    "humaneval": {
        "limit": 164,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "humaneval",
        "extra_args": " --trust_remote_code --confirm_run_unsafe_code",
    },
    "gsm8k": {
        "limit": 250,  # Reduced from full dataset to avoid problematic samples
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "gsm8k",
        "extra_args": "--num_fewshot 5",
    },
    "minerva_math_algebra": {
        "limit": 250,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "minerva_math_algebra",
        "extra_args": "--num_fewshot 4",
    },
    "truthfulqa": {
        "limit": 500,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "truthfulqa_gen",
        "extra_args": "--num_fewshot 0",
    },
    "truthfulqa_mc2": {
        "limit": 1,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "truthfulqa_mc2",
        "extra_args": "--num_fewshot 0",
    },
    "triviaqa": {
        "limit": 500,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "longbench_triviaqa",
        "extra_args": "--num_fewshot 0 --trust_remote_code --confirm_run_unsafe_code",
    },
    "mt_bench": {
        "limit": 1,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "mt_bench",
        "extra_args": "",
    },
    "mbpp": {
        "limit": 250,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "mbpp",
        "extra_args": " --trust_remote_code --confirm_run_unsafe_code --num_fewshot 3",
    },
    "hotpotqa": {
        "limit": 500,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "longbench_hotpotqa",
        "extra_args": "--num_fewshot 0 --trust_remote_code --confirm_run_unsafe_code",
    },
    "xsum": {
        "limit": 250,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "xsum",
        "extra_args": "--num_fewshot 0",
    },
    "cnn_dailymail": {
        "limit": 250,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "cnn_dailymail",
        "extra_args": "--num_fewshot 0",
    },
    "coqa": {
        "limit": 250,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "coqa",
        "extra_args": "--num_fewshot 0",
    },
    "longbench_narrativeqa": {
        "limit": 250,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "longbench_narrativeqa",
        "extra_args": "--num_fewshot 0",
    },
    "squad_completion": {
        "limit": 250,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "squad_completion",
        "extra_args": "--num_fewshot 0",
    },
    "squadv2": {
        "limit": 250,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "squadv2",
        "extra_args": "--num_fewshot 0",
    },
    # Add more benchmarks here if needed
}

terminate_flag = threading.Event()
thread_processes = {}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def is_multimodal_benchmark(benchmark: str) -> bool:
    return benchmark in {"chartqa"} or benchmark.startswith("mmmu")

def get_mixed_bmk_name(benchmarks):
    bmkname = ""
    for bmk in benchmarks:
        bmkname += f"{bmk}_"
    # remove the last underscore
    bmkname = bmkname[:-1]
    bmkname += f"_mixed"
    return bmkname

def curl_metrics(
    model,
    stat_filename,
    spec_decode="ngram",
    benchmarks=["gsm8k"],
    duration=60,
    k=None,
    server_address="localhost:8000",
    metrics_address=None
):
    # 1. Resolve default settings
    k = k if k is not None else 0
    maxexp = 8
    conf_thres = 1.0

    # 2. Build paths and filenames
    model_path = model.rstrip("/") if model.endswith("/") else model
    model_name = os.path.basename(model_path)

    bmkname = get_mixed_bmk_name(benchmarks)

    stats_dir = os.path.join(QUALITY_STATS_ROOT, bmkname, spec_decode,
                             model_name)
    os.makedirs(stats_dir, exist_ok=True)

    stat_file = f"{stat_filename}_n{duration}_k{k}_maxexp{maxexp}_thres{conf_thres}"
    metrics_target = metrics_address or server_address
    metrics_cmd = f"curl http://{metrics_target}/metrics > {stats_dir}/{stat_file}.metrics"
    subprocess.run(metrics_cmd, shell=True, check=True)

def run_spec_decode_eval(
    model,
    stat_filename,
    spec_decode="ngram",
    benchmark="humaneval",
    limit=None,
    k=None,
    maxexp=None,
    conf_thres=None,
    mt_bmk=None,
    conf_file=None,
    extra_args=None,
    thread_name=None,
    server_address="localhost:8000",
    metrics_address=None
):
    """
    Runs the chosen benchmark using 'lm-eval'. The directories, file names,
    and other logic now include the 'spec_decode' string (e.g., 'ngram').
    """
    if terminate_flag.is_set():
        return
    # 1. Resolve default settings
    cfg = bmk_defaults.get(benchmark)
    if cfg is None:
        cfg = {
            "limit": 250,
            "k": 0,
            "maxexp": 8,
            "conf_thres": 1.0,
            "tasks": benchmark,
            "extra_args": "",
        }
        print(
            f"Benchmark '{benchmark}' not found in defaults; using task name "
            f"'{benchmark}' directly.")
    limit = limit if limit is not None else cfg["limit"]
    k = k if k is not None else cfg["k"]
    maxexp = maxexp if maxexp is not None else cfg["maxexp"]
    conf_thres = conf_thres if conf_thres is not None else cfg["conf_thres"]
    if extra_args is None:
        extra_args = cfg.get("extra_args", "") or ""

    # 2. Build paths and filenames
    model_path = model.rstrip("/") if model.endswith("/") else model
    model_name = os.path.basename(model_path)
    conf_name = os.path.basename(conf_file) if conf_file else "default"
    conf_name = conf_name.replace(".json", "")

    stats_dir = os.path.join(QUALITY_STATS_ROOT, benchmark, spec_decode,
                             model_name)
    os.makedirs(stats_dir, exist_ok=True)

    stat_file = f"{stat_filename}_n{limit}_conf_{conf_name}"

    if benchmark.startswith("imo_answerbench"):
        dataset_name = os.environ.get("IMO_ANSWERBENCH_DATASET",
                                      "OpenEvals/IMO-AnswerBench")
        split = os.environ.get("IMO_ANSWERBENCH_SPLIT", "train")
        max_tokens = os.environ.get("IMO_ANSWERBENCH_MAX_TOKENS", "128")
        max_model_len = os.environ.get("IMO_ANSWERBENCH_MAX_MODEL_LEN",
                                       "4096")
        results_file = f"{stats_dir}/{stat_file}.results.jsonl"
        summary_file = f"{stats_dir}/{stat_file}.summary.json"
        imo_cmd = (
            f"python imo_answerbench_online.py "
            f"--server-address {server_address} "
            f"--model {model} "
            f"--dataset-name {dataset_name} "
            f"--split {split} "
            f"--max-tokens {max_tokens} "
            f"--max-model-len {max_model_len} "
            f"--output-file {results_file} "
            f"--summary-file {summary_file} "
            f"--limit {limit} "
            f"2>&1 | tee {stats_dir}/{stat_file}.log"
        )
        print(f"eval_cmd: {imo_cmd}")
        process = subprocess.Popen(
            imo_cmd,
            shell=True,
            preexec_fn=os.setsid,
        )
        if thread_name:
            thread_processes[thread_name] = process
    elif benchmark.startswith("swebench"):
        swebench_dataset_map = {
            "swebench": "SWE-bench/SWE-bench",
            "swebench_lite": "SWE-bench/SWE-bench_Lite",
            "swebench_verified": "SWE-bench/SWE-bench_Verified",
        }
        dataset_name = swebench_dataset_map.get(benchmark, "SWE-bench/SWE-bench")
        max_model_len = os.environ.get("MAX_MODEL_LEN", "4096")
        max_tokens = os.environ.get("SWE_BENCH_MAX_TOKENS", "4096")
        predictions_file = f"{stats_dir}/{stat_file}.predictions.jsonl"
        run_id = f"{stat_file}_run"
        swebench_cmd = (
            f"python swebench_harness_online.py "
            f"--server-address {server_address} "
            f"--model {model} "
            f"--dataset-name {dataset_name} "
            f"--split test "
            f"--predictions-file {predictions_file} "
            f"--run-id {run_id} "
            f"--max-model-len {max_model_len} "
            f"--max-tokens {max_tokens} "
            f"--limit {limit} "
            f"2>&1 | tee {stats_dir}/{stat_file}.log"
        )
        print(f"eval_cmd: {swebench_cmd}")
        process = subprocess.Popen(
            swebench_cmd,
            shell=True,
            preexec_fn=os.setsid,
        )
        if thread_name:
            thread_processes[thread_name] = process
    elif benchmark != "mt_bench":
        # 3. Build lm-eval command
        max_model_len = os.environ.get("MAX_MODEL_LEN", "4096")
        force_chat = env_flag("LM_EVAL_FORCE_CHAT_COMPLETIONS") or (
            "omni" in model.lower()
        )
        use_chat_completions = is_multimodal_benchmark(benchmark) or force_chat
        force_apply_chat_template = env_flag(
            "LM_EVAL_FORCE_APPLY_CHAT_TEMPLATE", default=force_chat)
        lm_eval_num_concurrent = os.environ.get("LM_EVAL_NUM_CONCURRENT", "16")
        lm_eval_timeout = os.environ.get("LM_EVAL_TIMEOUT")
        lm_eval_max_retries = os.environ.get("LM_EVAL_MAX_RETRIES")
        lm_eval_gen_kwargs = os.environ.get("LM_EVAL_GEN_KWARGS", "").strip()

        lm_eval_model = ("local-chat-completions"
                         if use_chat_completions else "local-completions")
        base_url_path = ("/v1/chat/completions"
                         if use_chat_completions else "/v1/completions")
        tokenizer_args = (
            ",tokenizer_backend=None,tokenized_requests=False"
            if use_chat_completions else "")
        model_args = (
            f"base_url=http://{server_address}{base_url_path},"
            "add_bos_token=True,"
            f"max_model_len={max_model_len},"
            f"max_length={max_model_len},"
            f"num_concurrent={lm_eval_num_concurrent}"
            f"{tokenizer_args}"
        )
        if lm_eval_timeout:
            model_args += f",timeout={lm_eval_timeout}"
        if lm_eval_max_retries:
            model_args += f",max_retries={lm_eval_max_retries}"

        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        if (use_chat_completions or force_apply_chat_template) and (
                "--apply_chat_template" not in extra_args):
            extra_args = f"{extra_args} --apply_chat_template".strip()

        gen_kwargs_arg = ""
        if lm_eval_gen_kwargs:
            gen_kwargs_arg = f"--gen_kwargs {shlex.quote(lm_eval_gen_kwargs)} "

        lm_eval_cmd = (
            f"lm-eval --model {lm_eval_model} "
            f"--tasks {cfg['tasks']} "
            f"--model_args model={model},{model_args} "
            f"--limit {limit} --log_samples {gen_kwargs_arg}{extra_args} "
            f"--output_path {stats_dir}/{stat_file}.jsonl "
            f"2>&1 | tee {stats_dir}/{stat_file}.log"
        )
        # subprocess.run(lm_eval_cmd, shell=True, check=True)
        print(f"eval_cmd: {lm_eval_cmd}")
        process = subprocess.Popen(
            lm_eval_cmd,
            shell=True,
            preexec_fn=os.setsid,
        )

        # Save the process associated with this thread
        if thread_name:
            thread_processes[thread_name] = process
    else:
        bmks = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]
        if mt_bmk is not None:
            print(f"Running MT-Bench bmk {mt_bmk} and ignoring limit")
            # find index of mt_bmk in bmks
            bmk_idx = bmks.index(mt_bmk)
            begin_idx = 10*bmk_idx
            end_idx = 10*(bmk_idx+1)
        else:
            # hardcoding extraction
            begin_idx = 50
            end_idx = 50 + limit
        fastchat_cmd = (
            f"""
            eval "$(conda shell.bash hook)" && \
            conda init bash && \
            conda activate fastchat && \
            export OPENAI_API_KEY="EMPTY" && \
            cd /home/asaxena317/moe-llm-restricted-sets/fastchat/fastchat/llm_judge && \
            python gen_api_answer.py \
                --model {model} \
                --openai-api-base http://{server_address}/v1 \
                --question-begin {begin_idx} --question-end {end_idx} \
                --force-temperature 0.0 \
                --answer-file {stats_dir}/{stat_file}.jsonl \
                2>&1 | tee {stats_dir}/{stat_file}.log
            """
        )
        # subprocess.run(fastchat_cmd, shell=True, executable="/bin/bash", check=True)

        process = subprocess.Popen(
            fastchat_cmd,
            shell=True,
            executable="/bin/bash",
            preexec_fn=os.setsid,
        )

        # Save the process associated with this thread
        if thread_name:
            thread_processes[thread_name] = process
    try:
        while True:
            if terminate_flag.is_set():
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)  # Force kill with SIGKILL (signal 9)
                break
            retcode = process.poll()
            if retcode is not None:  # Process completed
                break
            time.sleep(1)  # Check every second

    except KeyboardInterrupt:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()

    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    print(f"collecting metrics for {stat_file}")
    time.sleep(30)
    # 4. Grab metrics from server
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics_target = metrics_address or server_address
    metrics_cmd = f"curl http://{metrics_target}/metrics > {stats_dir}/{stat_file}_{timestamp}.metrics"
    subprocess.run(metrics_cmd, shell=True, check=True)

def run_benchmark_for_duration(args, model, stat_filename, 
                               benchmark, global_duration, thread_name=None):
    end_time = time.time() + global_duration
    runcount = 0
    while time.time() < end_time and not terminate_flag.is_set():
        runcount += 1
        curr_stat_filename = f"{stat_filename}_rc{runcount}"
        run_spec_decode_eval(
            model=model,
            stat_filename=curr_stat_filename,
            spec_decode=args.spec_decode,
            benchmark=benchmark,
            limit=args.limit,
            k=args.k,
            maxexp=args.maxexp,
            conf_thres=args.conf_thres,
            mt_bmk=args.mt_bmk,
            conf_file=args.config_file,
            extra_args=None,
            thread_name=thread_name,
            server_address=args.server_address,
            metrics_address=getattr(args, "metrics_address", None)
        )

def run_mixed_benchmarks(args, model, stat_filename, benchmarks, global_duration):
    threads = []
    stat_filename = f"mixed_{stat_filename}"
    try:
        for benchmark in benchmarks:
            thread_name = f"{benchmark}_thread"
            thread = threading.Thread(
                target=run_benchmark_for_duration, 
                args=(args, model, stat_filename, 
                      benchmark, global_duration, thread_name)
            )
            thread.start()
            threads.append(thread)
        start_time = time.time()
        while time.time() - start_time < global_duration:
            time.sleep(1)
        print("Terminating threads...")
        terminate_flag.set()
        for name, process in thread_processes.items():
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        time.sleep(3)
        for thread in threads:
            thread.join()

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected! Sending SIGKILL to all running benchmarks...")
        terminate_flag.set()
        for name, process in thread_processes.items():
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        for thread in threads:
            thread.join()
        print("All benchmarks forcibly terminated.")

    print("Threads terminated. Collecting mixed-bmk metrics...")
    curl_metrics(
        model=model,
        stat_filename=stat_filename,
        spec_decode=args.spec_decode,
        benchmarks=benchmarks,
        duration=global_duration,
        k=args.k,
        server_address=args.server_address,
        metrics_address=getattr(args, "metrics_address", None)
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model path or name.")
    parser.add_argument("-o", "--stat_filename", required=True, help="Base stat filename.")
    parser.add_argument("-b", "--benchmark", default=["gsm8k"],nargs='+',
                        help="Comma-separated list of benchmarks to run (e.g., gsm8k, humaneval).")
    parser.add_argument("-l", "--limit", type=float, default=None,
                        help="Number of samples to limit. If <1, limit is a percentage. (Default depends on benchmark.)")
    parser.add_argument("-k", "--k", type=int, default=None,
                        help="Parameter K. (Default depends on benchmark.)")
    parser.add_argument("-e", "--maxexp", type=int, default=8,
                        help="Parameter MAXEXP.")
    parser.add_argument("-c", "--conf_thres", type=float, default=1.0,
                        help="Confidence threshold.")
    parser.add_argument("-sd", "--spec_decode", default="ngram",
                        help="Which spec_decode/variant to use (e.g., 'ngram', etc.).")
    parser.add_argument("-d", "--draft_model", default=None, help="Draft model path.")

    # mixed bmk arguments
    parser.add_argument("-dur", "--duration", type=int, default=600,
                        help="Duration (in seconds) to run mixed benchmarks.")

    # Optional arguments
    parser.add_argument("-s", "--serving_script",
                        help="Script that launches the vLLM server (default is 'online_serving_<spec_decode>.sh').")
    parser.add_argument("-t", "--sleep_time", type=int, default=30,
                        help="Seconds to wait for vLLM to start before running the benchmark.")
    parser.add_argument("--startup-timeout", type=int, default=1800,
                        help="Additional seconds to wait (via health checks) for vLLM to become ready.")
    parser.add_argument("-mt", "--mt_bmk", default=None, help="Select bmk for MT-Bench.")
    parser.add_argument("-cf", "--config_file", 
                        default=f"/nethome/rdudala3/prowl/configs/qwen/qwen_do-nothing.json", 
                        help="Lynx config file.")
    parser.add_argument("-sa", "--server_address", 
                        default="localhost:8000",
                        help="Server address in format host:port (default: localhost:8000)")
    parser.add_argument("--metrics-address", 
                        default=None,
                        help="Address to scrape /metrics; defaults to server_address.")
    parser.add_argument("--health-address", 
                        default=None,
                        help="Address used for readiness checks; defaults to server_address.")
    parser.add_argument("-mb", "--max_batch_size", type=int, default=16,
                        help="Max batch size for vLLM.")
    parser.add_argument("--enable-expert-parallel", action="store_true",
                        help="Enable vLLM expert parallel mode when launching the server.")
    args = parser.parse_args()

    # If user didn't provide a --serving_script, build one dynamically
    # based on the spec_decode name:
    if not args.serving_script:
        args.serving_script = f"./online_serving_{args.spec_decode}.sh"

    # Derive default ports/addresses for metrics/readiness checks
    metrics_address = args.metrics_address or args.server_address
    health_address = args.health_address or args.server_address

    # Persist derived values back onto args for downstream functions
    args.metrics_address = metrics_address
    args.health_address = health_address

    bg_pid = None
    try:
        # ---------------------------------------------------
        # 1. Start vLLM serving in the background
        # ---------------------------------------------------
        if len(args.benchmark) > 1:
            bmkname = get_mixed_bmk_name(args.benchmark)
        else:
            bmkname = f"{args.benchmark[0]}_n{args.limit or 'def'}"  # might use the limit or 'def'
        vllm_statfilename = f"{bmkname}_{args.stat_filename}"
        if args.max_batch_size != 16:
            vllm_statfilename = f"{vllm_statfilename}_batch{args.max_batch_size}"
        print(f"vLLM statfile {vllm_statfilename}...")
        if args.spec_decode != "ngram" and args.draft_model is None:
            raise ValueError("Draft model path is required for non-ngram spec_decode variants.")
        print(f"Starting vLLM serving with script: {args.serving_script}")
        print(f"config_file: {args.config_file}")
        print(f"client address: {args.server_address}, metrics address: {metrics_address}, health address: {health_address}")
        # Extract port from server_address (e.g., "localhost:8000" -> "8000")
        port = args.server_address.split(':')[-1] if ':' in args.server_address else '8000'
        ep_token = "true" if args.enable_expert_parallel else "false"

        serving_cmd = (
            f"bash -c '{args.serving_script} {args.model} {vllm_statfilename} "
            f"{args.k if args.k is not None else 0} "
            f"{args.maxexp if args.maxexp is not None else 8} "
            f"{args.conf_thres if args.conf_thres is not None else 1.0} "
            f"{args.config_file} "
            f"{port} "
            f"{args.max_batch_size} "
            f"{ep_token} "
            f"> /dev/null 2>&1'"
        )
        print(f"serving_cmd: {serving_cmd}")
        processA = subprocess.Popen(serving_cmd, shell=True, start_new_session=True)
        bg_pid = processA.pid

        if args.sleep_time > 0:
            time.sleep(args.sleep_time)

        def wait_for_server_ready(address: str, timeout: int = 600, interval: int = 5):
            """Poll the /health endpoint until it responds or timeout expires."""
            deadline = time.time() + timeout
            health_url = f"http://{address}/health"
            while time.time() < deadline and not terminate_flag.is_set():
                try:
                    with urllib.request.urlopen(health_url, timeout=10) as resp:
                        if resp.status == 200:
                            print(f"vLLM server at {address} is healthy.")
                            return
                except urllib.error.URLError as exc:
                    print(f"Waiting for vLLM server at {address} to become ready: {exc}")
                time.sleep(interval)
            raise TimeoutError(f"Timed out waiting for vLLM server at {address} to become ready.")

        wait_for_server_ready(health_address, timeout=args.startup_timeout)

        # ---------------------------------------------------
        # 2. Run the Python benchmark in the foreground
        # ---------------------------------------------------
        if len(args.benchmark) == 1:
            run_spec_decode_eval(
                model=args.model,
                stat_filename=args.stat_filename + (f"_batch{args.max_batch_size}" if args.max_batch_size != 16 else ""),
                spec_decode=args.spec_decode,
                benchmark=args.benchmark[0],
                limit=args.limit,
                k=args.k,
                maxexp=args.maxexp,
                conf_thres=args.conf_thres,
                mt_bmk=args.mt_bmk,
                conf_file=args.config_file,
                server_address=args.server_address,
                metrics_address=args.metrics_address
            )
        else:
            print(f"Running mixed benchmarks: {args.benchmark} for {args.duration} seconds")
            run_mixed_benchmarks(args, args.model, args.stat_filename, args.benchmark, args.duration)
    except KeyboardInterrupt:
        print("Caught Ctrl+C (KeyboardInterrupt). Cleaning up...")

    finally:
        # ---------------------------------------------------
        # 3. Gracefully terminate vLLM if it was started
        # ---------------------------------------------------
        if bg_pid is not None:
            time.sleep(30)
            try:
                bg_pgid = os.getpgid(bg_pid)
                print(f"Killing process group {bg_pgid} with SIGINT...")
                os.killpg(bg_pgid, signal.SIGINT)
                time.sleep(120)
                os.killpg(bg_pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"Error while terminating background process group: {e}")

        print("Done.")

if __name__ == "__main__":
    main()
