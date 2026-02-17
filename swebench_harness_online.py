#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def extract_diff_fallback(response_text: str) -> str:
    if "```diff" in response_text:
        start = response_text.find("```diff")
        end = response_text.find("```", start + 7)
        if end != -1:
            return response_text[start + 7:end].strip()
    if "```patch" in response_text:
        start = response_text.find("```patch")
        end = response_text.find("```", start + 8)
        if end != -1:
            return response_text[start + 8:end].strip()
    if "```" in response_text:
        start = response_text.find("```")
        end = response_text.find("```", start + 3)
        if end != -1:
            return response_text[start + 3:end].strip()
    return response_text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n...[truncated]...\n"
    if max_chars <= len(marker) + 4:
        return text[:max_chars]
    head = int(max_chars * 0.7)
    tail = max_chars - head - len(marker)
    if tail <= 0:
        return text[:max_chars]
    return text[:head] + marker + text[-tail:]


def call_local_completions(server_address: str, model: str, prompt: str,
                           max_tokens: int, temperature: float,
                           max_model_len: int) -> str:
    # Conservative token estimate to avoid 400s from context overflow.
    est_input_tokens = max(1, len(prompt) // 3)
    allowed_tokens = max(64, max_model_len - est_input_tokens - 16)
    start_tokens = min(max_tokens, allowed_tokens)

    candidates = []
    for t in [start_tokens, 1024, 768, 512, 384, 256, 128, 64]:
        if t > 0 and t <= start_tokens and t not in candidates:
            candidates.append(t)

    last_error = None
    for candidate_max_tokens in candidates:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": candidate_max_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            f"http://{server_address}/v1/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("text", "")
        except urllib.error.HTTPError as err:
            body_text = ""
            try:
                body_text = err.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            last_error = RuntimeError(
                f"HTTP {err.code} for max_tokens={candidate_max_tokens}. "
                f"Body: {body_text[:1000]}")
            if err.code == 400 and "max_tokens" in body_text:
                continue
            if err.code == 400 and "maximum context length" in body_text:
                shortened = truncate_text(prompt, int(len(prompt) * 0.8))
                if shortened != prompt:
                    return call_local_completions(
                        server_address,
                        model,
                        shortened,
                        max_tokens,
                        temperature,
                        max_model_len,
                    )
            raise last_error

    if last_error is not None:
        raise last_error
    return ""


def build_prompt(problem_statement: str,
                 hints_text: str = "",
                 max_chars: int | None = None) -> str:
    prefix = (
        "You are a software engineering assistant.\n"
        "Given the issue below, produce ONLY a git patch that fixes it.\n"
        "Do not include explanations.\n\n"
        "Issue:\n"
    )
    suffix = "\nReturn a patch in unified diff format."

    problem = problem_statement
    hints = hints_text
    hint_block = f"\nHints:\n{hints}\n" if hints else "\n"
    prompt = f"{prefix}{problem}{hint_block}{suffix}"

    if max_chars is None or len(prompt) <= max_chars:
        return prompt

    if hints:
        over = len(prompt) - max_chars
        target = max(len(hints) - over, 0)
        hints = truncate_text(hints, target)
        hint_block = f"\nHints:\n{hints}\n" if hints else "\n"
        prompt = f"{prefix}{problem}{hint_block}{suffix}"

    if len(prompt) <= max_chars:
        return prompt

    over = len(prompt) - max_chars
    target = max(len(problem) - over, 0)
    problem = truncate_text(problem, target)
    prompt = f"{prefix}{problem}{hint_block}{suffix}"

    if len(prompt) > max_chars:
        return prompt[:max_chars]
    return prompt


def resolve_limit(total: int, limit_arg: str | None) -> int:
    if not limit_arg:
        return total
    try:
        limit_val = float(limit_arg)
    except ValueError:
        return total
    if limit_val <= 0:
        return total
    if limit_val < 1:
        return max(1, int(total * limit_val))
    return min(total, int(limit_val))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-address", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-name", default="SWE-bench/SWE-bench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--swebench-root",
                        default="/nethome/rdudala3/SWE-bench")
    args = parser.parse_args()

    swebench_root = Path(args.swebench_root)
    if not swebench_root.exists():
        raise FileNotFoundError(
            f"SWE-bench root not found at {swebench_root}. "
            "Set --swebench-root explicitly.")

    # Make SWE-bench importable for this process.
    sys.path.insert(0, str(swebench_root))

    from datasets import load_dataset  # type: ignore

    ds = load_dataset(args.dataset_name, split=args.split)
    num_to_run = resolve_limit(len(ds), args.limit)
    ds = ds.select(range(num_to_run))

    predictions_path = Path(args.predictions_file)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating predictions for {num_to_run} instances from "
        f"{args.dataset_name}:{args.split}"
    )

    with predictions_path.open("w", encoding="utf-8") as f:
        for datum in ds:
            instance_id = datum["instance_id"]
            max_prompt_chars = None
            if args.max_model_len:
                max_input_tokens = max(1, args.max_model_len - 80)
                max_prompt_chars = int(max_input_tokens * 2.7)
            prompt = build_prompt(
                datum.get("problem_statement", ""),
                datum.get("hints_text", ""),
                max_chars=max_prompt_chars,
            )
            completion = call_local_completions(
                args.server_address,
                args.model,
                prompt,
                args.max_tokens,
                args.temperature,
                args.max_model_len,
            )
            pred = {
                "instance_id": instance_id,
                "model_name_or_path": args.model,
                "model_patch": extract_diff_fallback(completion),
            }
            f.write(json.dumps(pred) + "\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{swebench_root}:{env.get('PYTHONPATH', '')}"
        if env.get("PYTHONPATH")
        else str(swebench_root)
    )

    eval_cmd = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        args.dataset_name,
        "--split",
        args.split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(args.max_workers),
        "--run_id",
        args.run_id,
    ]
    print("Running SWE-bench harness:", " ".join(eval_cmd))
    subprocess.run(eval_cmd, check=True, env=env, cwd=str(swebench_root))

    # Read harness summary report and append a final summary record.
    report_file = swebench_root / (
        args.model.replace("/", "__") + f".{args.run_id}.json"
    )
    if report_file.exists():
        with report_file.open("r", encoding="utf-8") as rf:
            report = json.load(rf)
        total_instances = int(report.get("total_instances", 0) or 0)
        submitted_instances = int(report.get("submitted_instances", 0) or 0)
        completed_instances = int(report.get("completed_instances", 0) or 0)
        resolved_instances = int(report.get("resolved_instances", 0) or 0)
        resolved_rate_total = (
            (resolved_instances / total_instances) if total_instances else 0.0
        )
        resolved_rate_completed = (
            (resolved_instances / completed_instances) if completed_instances else 0.0
        )
        summary_record = {
            "_type": "summary",
            "metric": "swebench_resolved_rate",
            "resolved_rate_total": resolved_rate_total,
            "resolved_rate_completed": resolved_rate_completed,
            "resolved_instances": resolved_instances,
            "completed_instances": completed_instances,
            "submitted_instances": submitted_instances,
            "total_instances": total_instances,
            "report_file": str(report_file),
        }
        with predictions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary_record) + "\n")
        print(
            f"SWE-bench resolved rate: {resolved_instances}/{total_instances} "
            f"= {resolved_rate_total:.4f}"
        )
        print(f"SWE-bench report: {report_file}")
        print(f"Summary appended to predictions: {predictions_path}")
    else:
        print(
            "WARNING: SWE-bench report file not found; could not append summary. "
            f"Expected: {report_file}"
        )


if __name__ == "__main__":
    main()
