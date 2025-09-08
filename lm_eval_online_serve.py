#!/usr/bin/env python3

import os
import time
import signal
import argparse
import subprocess
import threading

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
        "limit": 500,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "minerva_math_algebra",
        "extra_args": "--num_fewshot 4",
    },
    "truthfulqa_gen": {
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
    "mt_bench": {
        "limit": 1,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "mt_bench",
        "extra_args": "",
    },
    "mbpp": {
        "limit": 500,
        "k": 0,
        "maxexp": 8,
        "conf_thres": 1.0,
        "tasks": "mbpp",
        "extra_args": " --trust_remote_code --confirm_run_unsafe_code --num_fewshot 3",
    },
    # Add more benchmarks here if needed
}

terminate_flag = threading.Event()
thread_processes = {}

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
    server_address="localhost:8000"
):
    # 1. Resolve default settings
    k = k if k is not None else 0
    maxexp = 8
    conf_thres = 1.0

    # 2. Build paths and filenames
    model_path = model.rstrip("/") if model.endswith("/") else model
    model_name = os.path.basename(model_path)

    bmkname = get_mixed_bmk_name(benchmarks)

    stats_dir = os.path.expanduser(f"/nethome/vgupta345/stats/quality/{bmkname}/{spec_decode}/{model_name}/")
    os.makedirs(stats_dir, exist_ok=True)

    stat_file = f"{stat_filename}_n{duration}_k{k}_maxexp{maxexp}_thres{conf_thres}"
    metrics_cmd = f"curl http://{server_address}/metrics > {stats_dir}/{stat_file}.metrics"
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
    server_address="localhost:8000"
):
    """
    Runs the chosen benchmark using 'lm-eval'. The directories, file names,
    and other logic now include the 'spec_decode' string (e.g., 'ngram').
    """
    if terminate_flag.is_set():
        return
    # 1. Resolve default settings
    cfg = bmk_defaults.get(benchmark, bmk_defaults["humaneval"])
    limit = limit if limit is not None else cfg["limit"]
    k = k if k is not None else cfg["k"]
    maxexp = maxexp if maxexp is not None else cfg["maxexp"]
    conf_thres = conf_thres if conf_thres is not None else cfg["conf_thres"]
    extra_args = cfg["extra_args"] if cfg["extra_args"] else ""
    if extra_args is None:
        extra_args = ""

    # 2. Build paths and filenames
    model_path = model.rstrip("/") if model.endswith("/") else model
    model_name = os.path.basename(model_path)
    conf_name = os.path.basename(conf_file) if conf_file else "default"
    conf_name = conf_name.replace(".json", "")

    stats_dir = os.path.expanduser(f"/nethome/vgupta345/stats/quality/{benchmark}/{spec_decode}/{model_name}/")
    os.makedirs(stats_dir, exist_ok=True)

    stat_file = f"{stat_filename}_n{limit}_conf_{conf_name}"

    if benchmark != "mt_bench":
        # 3. Build lm-eval command
        model_args = (
            f"base_url=http://{server_address}/v1/completions,"
            "add_bos_token=True,"
            "max_model_len=4096,"
            "max_length=4096,"
            "num_concurrent=20,"
        )

        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        lm_eval_cmd = (
            f"lm-eval --model local-completions "
            f"--tasks {cfg['tasks']} "
            f"--model_args model={model},{model_args} "
            f"--limit {limit} --log_samples {extra_args} {extra_args} "
            f"--output_path {stats_dir}/{stat_file}.jsonl "
            f"--verbosity DEBUG "
            f"2>&1 | tee {stats_dir}/{stat_file}.log"
        )
        # subprocess.run(lm_eval_cmd, shell=True, check=True)
        print(f"eval_cmd: {lm_eval_cmd}")
        process = subprocess.Popen(lm_eval_cmd, shell=True, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   preexec_fn=os.setsid)

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
                --openai-api-base http://{args.server_address}/v1 \
                --question-begin {begin_idx} --question-end {end_idx} \
                --force-temperature 0.0 \
                --answer-file {stats_dir}/{stat_file}.jsonl \
                2>&1 | tee {stats_dir}/{stat_file}.log
            """
        )
        # subprocess.run(fastchat_cmd, shell=True, executable="/bin/bash", check=True)

        process = subprocess.Popen(fastchat_cmd, shell=True, 
                                   executable="/bin/bash",
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   preexec_fn=os.setsid)

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
    metrics_cmd = f"curl http://{server_address}/metrics > {stats_dir}/{stat_file}.metrics"
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
            server_address=args.server_address
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
        k=args.k
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
    parser.add_argument("-mt", "--mt_bmk", default=None, help="Select bmk for MT-Bench.")
    parser.add_argument("-cf", "--config_file", 
                        default=f"/nethome/vgupta345/prowl-plots/configs/qwen/qwen_do-nothing.json", 
                        help="Lynx config file.")
    parser.add_argument("-sa", "--server_address", 
                        default="localhost:8000",
                        help="Server address in format host:port (default: localhost:8000)")
    args = parser.parse_args()

    # If user didn't provide a --serving_script, build one dynamically
    # based on the spec_decode name:
    if not args.serving_script:
        args.serving_script = f"./online_serving_{args.spec_decode}.sh"

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
        print(f"vLLM statfile {vllm_statfilename}...")
        if args.spec_decode != "ngram" and args.draft_model is None:
            raise ValueError("Draft model path is required for non-ngram spec_decode variants.")
        print(f"Starting vLLM serving with script: {args.serving_script}")
        print(f"config_file: {args.config_file}")
        # Extract port from server_address (e.g., "localhost:8000" -> "8000")
        port = args.server_address.split(':')[-1] if ':' in args.server_address else '8000'
        serving_cmd = (
            f"bash -c '{args.serving_script} {args.model} {vllm_statfilename} "
            f"{args.k if args.k is not None else 0} "
            f"{args.maxexp if args.maxexp is not None else 8} "
            f"{args.conf_thres if args.conf_thres is not None else 1.0} "
            f"{args.config_file} "
            f"{port} "
            f"> /dev/null 2>&1'"
        )
        print(f"serving_cmd: {serving_cmd}")
        processA = subprocess.Popen(serving_cmd, shell=True, start_new_session=True)
        bg_pid = processA.pid

        time.sleep(args.sleep_time)

        # ---------------------------------------------------
        # 2. Run the Python benchmark in the foreground
        # ---------------------------------------------------
        if len(args.benchmark) == 1:
            run_spec_decode_eval(
                model=args.model,
                stat_filename=args.stat_filename,
                spec_decode=args.spec_decode,
                benchmark=args.benchmark[0],
                limit=args.limit,
                k=args.k,
                maxexp=args.maxexp,
                conf_thres=args.conf_thres,
                mt_bmk=args.mt_bmk,
                conf_file=args.config_file,
                server_address=args.server_address
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
