import json
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_gen import model_generator

class Mossmoon_generator(model_generator):
    def __init__(self, f_path, is_few_shot, device, is_vllm, model_path, model_name, few_shot_path, tensor_parallel_size, gpu_memory_utilization):
        '''
        Args:
            model_name: The name of the model
        '''
        super(Mossmoon_generator, self).__init__(f_path, is_few_shot, device, is_vllm, model_path, few_shot_path, tensor_parallel_size, gpu_memory_utilization)
        self.model_name = model_name
        
    def model_init(self):
        '''
        Init the model
        '''
        if self.is_vllm:
            model, tokenizer = self.model_init_vllm()
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16).eval()
        return model, tokenizer
                
    def generate_output(self, q_type, tokenizer, model, batch_size=None):
        '''
        Generate the output for Moss_moon_sft, using the suitable prompt template
        '''
        task_name = self.f_path.split("/")[-1].split(".")[0]
        instruction_ls, input_text_ls, answer_ls = self.process_prompt(task_name)
        output_ls = []
        num_unsuccess = 0
        for instruction, input_text, answer in tqdm(zip(instruction_ls, input_text_ls, answer_ls), total=len(input_text_ls)):
            prompt = instruction + input_text
            prompt = "\n<|Human|>: " + prompt + "<eoh>\n<|MOSS|>:"
            # Truncate the prompt
            if task_name == "3_1" or task_name == "3_2" or task_name == "4_1":
                prompt = model_generator.truncate_long(prompt, 2028, tokenizer, q_type)
            else: 
                prompt = model_generator.truncate_long(prompt, 2048, tokenizer, q_type)
            inputs = tokenizer(prompt, return_tensors="pt")
            try:
                inputs = inputs.to(model.device)        
                outputs = model.generate(**inputs, do_sample=False, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id) if q_type == "multiple_choice" else model.generate(**inputs, do_sample=False, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            except:
                num_unsuccess += 1
                print("Fail to answer")
                response = "未成功回答"
            output_ls.append({"input": input_text,
                            "output": response,
                            "answer": answer})
        return output_ls, num_unsuccess