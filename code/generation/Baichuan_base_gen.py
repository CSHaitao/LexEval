import json
from typing import Dict
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_gen import model_generator

class Baichuan_base_generator(model_generator):
    def __init__(self, f_path, is_few_shot, device, is_vllm, model_path, model_name, few_shot_path, tensor_parallel_size, gpu_memory_utilization):
        '''
        Args:
            model_name: The name of the model
        '''
        super(Baichuan_base_generator, self).__init__(f_path, is_few_shot, device, is_vllm, model_path, few_shot_path, tensor_parallel_size, gpu_memory_utilization)
        self.model_name = model_name
        
    def model_init(self):
        '''
        Init the model
        '''
        if self.is_vllm:
            model, tokenizer = self.model_init_vllm()
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().eval().cuda()
        return model, tokenizer
                
    def generate_output(self, q_type, tokenizer, model, batch_size=None):
        '''
        Generate the output for baichuan_base, using the suitable prompt template
        '''
        task_name = self.f_path.split("/")[-1].split(".")[0]
        instruction_ls, input_text_ls, answer_ls = self.process_prompt(task_name)
        output_ls = []
        num_unsuccess = 0
        for instruction, input_text, answer in tqdm(zip(instruction_ls, input_text_ls, answer_ls), total=len(input_text_ls)):
            prompt = instruction + input_text
            # Truncate the prompt
            prompt = model_generator.truncate_long(prompt, 4096, tokenizer, q_type)
            inputs = tokenizer(prompt, return_tensors='pt')
            inputs = inputs.to(model.device)
            try:
                out = model.generate(**inputs, max_new_tokens=20, do_sample=False) if q_type == "multiple_choice" else model.generate(**inputs, max_new_tokens=512, do_sample=False)
                response = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]
            except:
                num_unsuccess += 1
                print("Fail to answer")
                response = "未成功回答"
            output_ls.append({"input": input_text,
                            "output": response,
                            "answer": answer})
        return output_ls, num_unsuccess