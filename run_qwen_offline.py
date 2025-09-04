import argparse

from vllm import LLM, SamplingParams
            
def main():
    batch_size = 16
    tp_size = 2

    #########################################################

    gpu_memory_utilization = 0.8
    max_model_len = 4096

    llm = LLM(model="Qwen/Qwen2-57B-A14B-Instruct",
              max_num_seqs=batch_size,
              swap_space=0,
              gpu_memory_utilization=gpu_memory_utilization,
              enable_chunked_prefill=False,
              max_model_len=max_model_len,
              tensor_parallel_size=tp_size,
              mixtral_config_file="/var/tmp/jae/prowl/configs/qwen/qwen_do-nothing.json")

    prompts = [f"{i}.once upon a time, there were {i} " for i in range(2*batch_size)]

    # llm.start_profile()
    outputs = llm.generate(prompts, 
                    sampling_params=SamplingParams(
                    temperature=0,
                    ignore_eos=False,
                    max_tokens=max_model_len))
    # llm.stop_profile()

    for output in outputs[:2]:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(len(output.outputs))
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    main()