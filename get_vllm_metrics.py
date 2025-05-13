import sys
import re

def parse_metrics(file_path):
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r'^(\S+)\{.*\}\s+([\d.e+-]+)$', line.strip())
            if match:
                key, value = match.groups()
                metrics[key] = float(value)
    return metrics

def compute_metrics(file_path):
    metrics = parse_metrics(file_path)
    
    gen_tokens = metrics.get("vllm:request_generation_tokens_sum", 0)
    decode_time = metrics.get("vllm:request_decode_time_seconds_sum", 1)  # Avoid division by zero
    total_gen_tokens = metrics.get("vllm:generation_tokens_total", 1)  # Avoid division by zero
    
    gen_throughput = gen_tokens / decode_time
    eff_tok_rate = gen_tokens / total_gen_tokens
    print(f"gen_tokens: {gen_tokens:.2f}")
    print(f"total_gen_tokens: {total_gen_tokens:.2f}")
    print(f"gen_throughput: {gen_throughput:.2f}")
    print(f"eff_tok_rate: {eff_tok_rate:.2f}")
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    compute_metrics(file_path)
