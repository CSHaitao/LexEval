import json
from tqdm import tqdm
import openai
import tiktoken
from model_gen import model_generator

class Chatgpt_generator(model_generator):
    def __init__(self, f_path, is_few_shot, device, api_base, api_key, few_shot_path, model_name):
        '''
        Args:
        api_base: openai api base
        api_key: openai api key
        '''
        super(Chatgpt_generator, self).__init__(f_path, is_few_shot, device, few_shot_path)
        openai.api_base = api_base
        openai.api_key = api_key
        self.model_name = model_name
                
    def generate_output(self, q_type, tokenizer=None, model=None):
        '''
        Generate the output for chatglm, using the suitable prompt template
        '''
        task_name = self.f_path.split("/")[-1].split(".")[0]
        instruction_ls, input_text_ls, answer_ls = self.process_prompt(task_name)
        output_ls = []
        num_unsuccess = 0
        for instruction, input_text, answer in tqdm(zip(instruction_ls, input_text_ls, answer_ls), total=len(input_text_ls)):
            prompt = instruction + input_text
            encoding = tiktoken.encoding_for_model(self.model_name)
            mark_prompt = encoding.encode(prompt)
            num_tokens = len(list(mark_prompt))
            # Truncate the prompt
            if num_tokens > 4090:
                print(f"Input tokens too long, cut to 4090 tokens!")
                mark_prompt = mark_prompt[0:2045]+mark_prompt[-2044:]
                prompt = encoding.decode(mark_prompt)
            chat_history = [{"role": "user", "content": prompt}]
            try:
                res = openai.ChatCompletion.create(
                                    model=self.model_name,
                                    messages=chat_history,
                                    temperature=0.0
                                )
                response = res["choices"][0]["message"]["content"]
            except:
                num_unsuccess += 1
                print("Fail to answer")
                response = "未成功回答"
            output_ls.append({"input": input_text,
                              "output": response,
                              "answer": answer})
        return output_ls, num_unsuccess