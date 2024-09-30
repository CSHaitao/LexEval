import json
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import LlamaForCausalLM, LlamaTokenizer
from model_gen import model_generator

class Lawgpt_generator(model_generator):
    def __init__(self, f_path, is_few_shot, device, is_vllm, model_path, model_name, few_shot_path, tensor_parallel_size, gpu_memory_utilization):
        '''
        Args:
            model_name: The name of the model
        '''
        super(Lawgpt_generator, self).__init__(f_path, is_few_shot, device, is_vllm, model_path, few_shot_path, tensor_parallel_size, gpu_memory_utilization)
        self.model_name = model_name
        
    def model_init(self):
        '''
        Init the model
        '''
        if self.is_vllm:
            model, tokenizer = self.model_init_vllm()
        else:
            tokenizer = LlamaTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            model = LlamaForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().cuda()
            model = model.eval()
        return model, tokenizer
                
    def generate_output(self, q_type, tokenizer, model, batch_size=None):
        '''
        Generate the output for Lawgpt, using the suitable prompt template
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
                    # For few-shot setting, not using prompt template since the performance without template is better
                    current_prompt = instruction[i] + input_text[i]
                    if not self.is_few_shot:
                        current_prompt = "你是中国顶尖智能法律顾问 LaWGPT，具备强大的中文法律基础语义理解能力，能够出色地理解和执行与法律问题和指令。你只能回答与中国法律领域相关的问题，其余领域的问题请礼貌地拒绝回答。接下来，请依据中国法律来回答下面这个问题。\n### 问题:\n" + current_prompt + "\n### 回答:\n"
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
                # For few-shot setting, not using prompt template since the performance without template is better
                if not self.is_few_shot:
                    prompt = "你是中国顶尖智能法律顾问 LaWGPT，具备强大的中文法律基础语义理解能力，能够出色地理解和执行与法律问题和指令。你只能回答与中国法律领域相关的问题，其余领域的问题请礼貌地拒绝回答。接下来，请依据中国法律来回答下面这个问题。\n### 问题:\n" + prompt + "\n### 回答:\n"
                # Truncate the prompt
                prompt = model_generator.truncate_long(prompt, 2048, tokenizer, q_type)
                inputs = tokenizer(prompt, return_tensors='pt')
                inputs = inputs.to(model.device)
                try:
                    with torch.no_grad():
                        pred = model.generate(**inputs, max_new_tokens=20, do_sample=False) if q_type == 'multiple_choice' else model.generate(**inputs, max_new_tokens=512, do_sample=False)
                        response = tokenizer.decode(pred[0], skip_special_tokens=True)[len(prompt):]
                except:
                    num_unsuccess += 1
                    print("Fail to answer")
                    response = "未成功回答" 
                output_ls.append({"input": input_text,
                                "output": response,
                                "answer": answer})
        return output_ls, num_unsuccess