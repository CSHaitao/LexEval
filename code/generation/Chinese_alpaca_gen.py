import json
import torch
from typing import Dict
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_gen import model_generator

class Chinese_alpaca_generator(model_generator):
    def __init__(self, f_path, is_few_shot, device, is_vllm, model_path, model_name, few_shot_path, tensor_parallel_size, gpu_memory_utilization):
        '''
        Args:
            model_name: The name of the model
        '''
        super(Chinese_alpaca_generator, self).__init__(f_path, is_few_shot, device, is_vllm, model_path, few_shot_path, tensor_parallel_size, gpu_memory_utilization)
        self.model_name = model_name
        
    def model_init(self):
        '''
        Init the model
        '''
        if self.is_vllm:
            model, tokenizer = self.model_init_vllm()
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True, device_map='auto').half().eval()
        return model, tokenizer
                
    def generate_output(self, q_type, tokenizer, model, batch_size=None):
        '''
        Generate the output for Chinese_alpaca, using the suitable prompt template
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
                    # Adding template will degrade the performance, so we don't use it
                    # system_prompt = """You are a helpful assistant. 你是一个乐于助人的助手。"""
                    # TEMPLATE = (
                    #             "[INST] <<SYS>>\n"
                    #             "{system_prompt}\n"
                    #             "<</SYS>>\n\n"
                    #             "{instruction} [/INST]"
                    #             )
                    # current_prompt = TEMPLATE.format_map({'instruction': current_prompt,'system_prompt': system_prompt})
                    prompt.append(current_prompt)
                # Truncate the prompt
                prompt = [model_generator.truncate_long(prompt[i], 4096, tokenizer, q_type) for i in range(len(prompt))]
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
                # system_prompt = """You are a helpful assistant. 你是一个乐于助人的助手。"""
                # TEMPLATE = (
                #             "[INST] <<SYS>>\n"
                #             "{system_prompt}\n"
                #             "<</SYS>>\n\n"
                #             "{instruction} [/INST]"
                #             )
                # prompt = TEMPLATE.format_map({'instruction': prompt,'system_prompt': system_prompt})
                # Truncate the prompt
                prompt = model_generator.truncate_long(prompt, 4096, tokenizer, q_type)
                inputs = tokenizer(prompt, return_tensors='pt')
                try:
                    if q_type == "multiple_choice":
                        generation_output = model.generate(
                                            input_ids = inputs["input_ids"].to(model.device), 
                                            attention_mask = inputs['attention_mask'].to(model.device),
                                            eos_token_id=tokenizer.eos_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            max_new_tokens=20,
                                            do_sample=False)
                    else:
                        generation_output = model.generate(
                                            input_ids = inputs["input_ids"].to(model.device),
                                            attention_mask = inputs['attention_mask'].to(model.device),
                                            eos_token_id=tokenizer.eos_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            max_new_tokens=512,
                                            do_sample=False)
                    response = tokenizer.decode(generation_output[0], skip_special_tokens=True)[len(prompt):]
                except:
                    num_unsuccess += 1
                    print("Fail to answer")
                    response = "未成功回答"
                output_ls.append({"input": input_text,
                                "output": response,
                                "answer": answer})
        return output_ls, num_unsuccess