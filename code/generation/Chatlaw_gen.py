import json
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_gen import model_generator

class Chatlaw_generator(model_generator):
    def __init__(self, f_path, is_few_shot, device, is_vllm, model_path, model_name, few_shot_path, tensor_parallel_size, gpu_memory_utilization, model_path_base):
        '''
        Args:
            model_name: The name of the model
        '''
        super(Chatlaw_generator, self).__init__(f_path, is_few_shot, device, is_vllm, model_path, few_shot_path, tensor_parallel_size, gpu_memory_utilization)
        self.model_name = model_name
        self.model_path_base = model_path_base
        
    def model_init(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path_base, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_path_base, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(model, self.model_path).eval()
        return model, tokenizer
                
    def generate_output(self, q_type, tokenizer, model, batch_size=None):
        '''
        Generate the output for Chatlaw, using the suitable prompt template
        '''
        task_name = self.f_path.split("/")[-1].split(".")[0]
        instruction_ls, input_text_ls, answer_ls = self.process_prompt(task_name)
        output_ls = []
        num_unsuccess = 0
        for instruction, input_text, answer in tqdm(zip(instruction_ls, input_text_ls, answer_ls), total=len(input_text_ls)):
            prompt = instruction + input_text
            prompt = f"Consult:\n{prompt}\nResponse:\n"
            # Truncate the prompt
            prompt = model_generator.truncate_long(prompt, 2048, tokenizer, q_type)
            inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
            inputs['input_ids'] = inputs['input_ids'].to(model.device)
            try:
                with torch.no_grad():
                    generation_output = model.generate(**inputs, max_new_tokens=20, do_sample=False) if q_type == "multiple_choice" else model.generate(**inputs, max_new_tokens=512, do_sample=False)
                    s = generation_output[0]
                    output = tokenizer.decode(s, skip_special_tokens=True)
                    response = output.split("Response:")[-1]
            except:
                num_unsuccess += 1
                print("Fail to answer")
                response = "未成功回答"
            output_ls.append({"input": input_text,
                              "output": response,
                              "answer": answer})
        return output_ls, num_unsuccess