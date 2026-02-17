from vllm import LLM, SamplingParams
            
def main():
    batch_size = 16
    tp_size = 4

    #########################################################

    gpu_memory_utilization = 0.9
    max_model_len = 4096

    llm = LLM(model="Qwen/Qwen3-235B-A22B-Instruct-2507",
              max_num_seqs=batch_size,
              swap_space=0,
              gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len,
              tensor_parallel_size=tp_size,
              mixtral_config_file="/nethome/rdudala3/prowl/configs/qwen3/quant_alpha1_optimized.json")

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
