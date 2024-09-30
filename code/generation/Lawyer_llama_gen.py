import json
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import LlamaForCausalLM, LlamaTokenizer
from model_gen import model_generator

class Lawyer_llama_generator(model_generator):
    def __init__(self, f_path, is_few_shot, device, is_vllm, model_path, model_name, few_shot_path, tensor_parallel_size, gpu_memory_utilization):
        '''
        Args:
            model_name: The name of the model
        '''
        super(Lawyer_llama_generator, self).__init__(f_path, is_few_shot, device, is_vllm, model_path, few_shot_path, tensor_parallel_size, gpu_memory_utilization)
        self.model_name = model_name
        
    def model_init(self):
        '''
        Init the model
        '''
        if self.is_vllm:
            model, tokenizer = self.model_init_vllm()
        else:
            tokenizer = LlamaTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            model = LlamaForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().eval().cuda()
        return model, tokenizer
                
    def generate_output(self, q_type, tokenizer, model, batch_size=None):
        '''
        Generate the output for Lawyer_llama, using the suitable prompt template
        '''
        task_name = self.f_path.split("/")[-1].split(".")[0]
        instruction_ls, input_text_ls, answer_ls = self.process_prompt(task_name)
        output_ls = []
        num_unsuccess = 0
        if self.is_vllm:
            sampling_params = SamplingParams(temperature=0, max_tokens=20) if q_type == 'multiple_choice' else SamplingParams(temperature=0, max_tokens=512)
            for idx in range(0, len(instruction_ls), batch_size):
                instruction, input_text, answer = instruction_ls[idx:idx+batch_size], input_text_ls[idx:idx+batch_size], answer_ls[idx:idx+batch_size]
                prompt = []
                for i in range(len(instruction)):
                    current_prompt = instruction[i] + input_text[i]
                    # Adding prompt template will degrade the model's performance, so we remove it in few-shot setting
                    if not self.is_few_shot:
                        system_prompt = "你是人工智能法律助手“Lawyer LLaMA”，能够回答与中国法律相关的问题。\n"
                        current_prompt = system_prompt + f"### Human: {current_prompt}\n### Assistant: "
                    prompt.append(current_prompt)
                # Truncate the prompt
                prompt = [model_generator.truncate_long(prompt[i], 2048, tokenizer, q_type) for i in range(len(prompt))]
                try:
                    response_ls = model.generate(prompt, sampling_params)
                    for i, out in enumerate(response_ls):
                        response = out.outputs[0].text
                        output_ls.append({"input": input_text[i],
                                    "output": response,
                                    "answer": answer[i]})
                except:
                    for i in range(len(instruction)):
                        num_unsuccess += 1
                        print("Fail to answer")
                        response = "未成功回答"
                        output_ls.append({"input": input_text,
                                  "output": response,
                                  "answer": answer})
        else:
            for instruction, input_text, answer in tqdm(zip(instruction_ls, input_text_ls, answer_ls), total=len(input_text_ls)):
                prompt = instruction + input_text
                # Adding prompt template will degrade the model's performance, so we remove it in few-shot setting
                if not self.is_few_shot:
                    system_prompt = "你是人工智能法律助手“Lawyer LLaMA”，能够回答与中国法律相关的问题。\n"
                    prompt = system_prompt + f"### Human: {prompt}\n### Assistant: "
                # Truncate the prompt
                prompt = model_generator.truncate_long(prompt, 2048, tokenizer, q_type)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
                try:
                    outputs = model.generate(input_ids, max_new_tokens=20, do_sample=False, use_cache=True) if q_type == "multiple_choice" else model.generate(input_ids, max_new_tokens=512, do_sample=False, use_cache=True)
                    output_text = str(tokenizer.decode(outputs[0], skip_special_tokens=True))
                    response = output_text.split("### Assistant:")[-1].strip()
                except:
                    num_unsuccess += 1
                    print("Fail to answer")
                    response = "未成功回答"
                output_ls.append({"input": input_text,
                                "output": response,
                                "answer": answer})
        return output_ls, num_unsuccess