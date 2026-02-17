#!/usr/bin/env python3

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path


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


def build_prompt(problem: str) -> str:
    return (
        "Solve the following IMO-level math problem.\n"
        "Return only the final short answer in one line.\n"
        "Do not include explanations.\n\n"
        f"Problem:\n{problem}\n"
        "\nFinal Answer: "
    )


def extract_boxed(text: str) -> str | None:
    match = re.findall(r"\\boxed\{([^}]*)\}", text)
    if match:
        return match[-1].strip()
    return None


def normalize_answer(text) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    if not t:
        return ""
    boxed = extract_boxed(t)
    if boxed is not None:
        t = boxed
    lines = t.splitlines()
    if lines:
        t = lines[-1].strip()
    else:
        t = t.strip()
    t = re.sub(r"^answer\s*[:=]\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^thefinalansweris\s*", "", t, flags=re.IGNORECASE)
    t = t.replace("$", "")
    t = t.rstrip(".")
    t = "".join(t.split())
    return t.lower()


def pick_first(datum: dict, keys: list[str]) -> str:
    for key in keys:
        if key in datum and datum[key] not in (None, ""):
            return str(datum[key])
    for key in keys:
        if key in datum:
            return "" if datum[key] is None else str(datum[key])
    return ""


def extract_final_answer(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    if not t:
        return ""

    boxed = extract_boxed(t)
    if boxed is not None:
        return boxed.strip()

    m = re.search(r"final answer\s*[:=]\s*(.*)", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).splitlines()[0].strip()

    m = re.search(r"answer\s*[:=]\s*(.*)", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).splitlines()[0].strip()

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[-1]


def call_local_completions(server_address: str, model: str, prompt: str,
                           max_tokens: int, temperature: float,
                           max_model_len: int) -> str:
    est_input_tokens = max(1, len(prompt) // 3)
    allowed_tokens = max(32, max_model_len - est_input_tokens - 16)
    start_tokens = min(max_tokens, allowed_tokens)
    candidates = []
    for t in [start_tokens, 384, 256, 192, 128, 96, 64, 48, 32]:
        if t > 0 and t <= start_tokens and t not in candidates:
            candidates.append(t)

    last_error = None
    for candidate_max_tokens in candidates:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": candidate_max_tokens,
            "temperature": temperature,
            "stop": ["\n"],
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
            raise last_error
    if last_error is not None:
        raise last_error
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-address", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-name", default="OpenEvals/IMO-AnswerBench")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    args = parser.parse_args()

    from datasets import load_dataset  # type: ignore

    ds = load_dataset(args.dataset_name, split=args.split)
    num_to_run = resolve_limit(len(ds), args.limit)
    ds = ds.select(range(num_to_run))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = 0
    with output_path.open("w", encoding="utf-8") as f:
        for i, datum in enumerate(ds):
            problem = pick_first(datum, [
                "problem",
                "Problem",
                "question",
                "Question",
                "prompt",
                "Prompt",
            ])
            reference = pick_first(datum, [
                "answer",
                "Answer",
                "short_answer",
                "Short Answer",
                "final_answer",
                "Final Answer",
                "solution",
                "Solution",
                "target",
                "Target",
            ])
            problem_id = pick_first(datum, [
                "Problem ID",
                "problem_id",
                "id",
                "ID",
            ])
            if i == 0 and (not problem or not reference):
                print(
                    "Warning: first sample has empty problem/reference. "
                    f"Available keys: {sorted(datum.keys())}"
                )
            completion = call_local_completions(
                args.server_address,
                args.model,
                build_prompt(problem),
                args.max_tokens,
                args.temperature,
                args.max_model_len,
            )
            extracted = extract_final_answer(completion)
            pred_norm = normalize_answer(extracted)
            ref_norm = normalize_answer(reference)
            is_correct = pred_norm == ref_norm
            correct += int(is_correct)
            total += 1
            f.write(json.dumps({
                "index": i,
                "problem_id": problem_id,
                "problem": problem,
                "reference_answer": reference,
                "prediction_raw": completion,
                "prediction_extracted": extracted,
                "prediction_norm": pred_norm,
                "reference_norm": ref_norm,
                "correct": is_correct,
            }) + "\n")

    acc = (correct / total) if total else 0.0
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": args.dataset_name,
            "split": args.split,
            "model": args.model,
            "num_samples": total,
            "num_correct": correct,
            "accuracy": acc,
        }, f, indent=2)

    print(f"IMO-AnswerBench done: {correct}/{total} correct (accuracy={acc:.4f})")
    print(f"Results: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
