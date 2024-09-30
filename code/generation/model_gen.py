import json
from vllm import LLM, SamplingParams

class model_generator:
    def __init__(self, f_path, is_few_shot, device, is_vllm, model_path, few_shot_path=None, tensor_parallel_size=None, gpu_memory_utilization=None):
        '''
        Args:
            f_path: The path for original data file in json format
            is_few_shot: Whether to utilize few-shot setting
            device: device number of cuda
            model_path: The path of the model
            few_shot_path: The path for few-shot data file
            tensor_parallel_size: Numbers of GPUs to use
            gpu_memory_utilization: Proportion of memory to use
        '''
        self.f_path = f_path
        self.model_path = model_path
        self.is_few_shot = is_few_shot
        if is_few_shot and few_shot_path == None:
            raise ValueError("Cannot find few-shot path")
        self.few_shot_path = few_shot_path
        self.device = device
        if is_vllm and (tensor_parallel_size == None or gpu_memory_utilization == None):
            raise ValueError("Please provide the tensor_parallel_size and gpu_memory_utilization, current value is none")
        self.is_vllm = is_vllm
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        
    def model_init_vllm(self):
        '''
        Init the model using vllm
        '''
        model = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size, 
            gpu_memory_utilization=self.gpu_memory_utilization, 
            trust_remote_code=True,
            dtype="half", # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True
        )
        tokenizer = model.get_tokenizer()
        return model, tokenizer
        
    def get_fewshot_examples(self, task_name):
        '''
        Get the few shot examples and gather them together
        '''
        with open(self.few_shot_path, 'r') as f:
            lines = f.readlines()
        examples = "以下是三个例子:\n"
        for line in lines:
            ex = json.loads(line)
            if task_name == "5_1" or task_name == "5_1_few_shot":
                examples = f"{examples}问题: {ex['input']}\n摘要: {ex['answer']}\n\n"
            elif task_name == "5_2" or task_name == "5_2_few_shot":
                examples = f"{examples}问题: {ex['input']}\n裁判分析过程: {ex['answer']}\n\n"
            elif task_name == "5_3" or task_name == "5_3_few_shot":
                examples = f"{examples}问题: {ex['input']}\n翻译结果: {ex['answer']}\n\n"
            else:
                examples = f"{examples}问题: {ex['input']}\n答案: {ex['answer']}\n\n"
        examples += "请你回答:\n问题:"
        return examples
    
    @staticmethod
    def truncate_long(prompt, context_length, tokenizer, q_type):
        '''
        For question with long context, truncate it.
        Args:
            prompt: Original prompt for the model
            context_length: Maximum context length for the model
            tokenizer: tokenizer for the model
            q_type: Must be 'generation' or 'multiple_choice'
        '''
        ori_prompt = tokenizer.encode(prompt)
        if q_type == 'generation':
            if len(ori_prompt) > context_length-512:
                print(f"Input tokens too long, cut to {context_length-512} tokens!")
                half = int((context_length-512)/2)
                prompt = tokenizer.decode(ori_prompt[:half], skip_special_tokens=True)+tokenizer.decode(ori_prompt[-half:], skip_special_tokens=True)
        elif q_type == 'multiple_choice':
            if len(ori_prompt) > context_length-20:
                print(f"Input tokens too long, cut to {context_length-20} tokens!")
                half = int((context_length-20)/2)
                prompt = tokenizer.decode(ori_prompt[:half], skip_special_tokens=True)+tokenizer.decode(ori_prompt[-half:], skip_special_tokens=True)
        else:
            raise ValueError(f"Wrong question type, q_type must be 'generation' or 'multiple_choice' but get {q_type}")
        return prompt
                
    def process_prompt(self, task_name):
        '''
        Concatenate the instruction, input, few-shot examples to get the prompt, and load the prompt in batches
        '''
        with open(self.f_path) as f1:
            lines = f1.readlines()
        instruction_ls, input_text_ls, answer_ls = [], [], []
        for idx, line in enumerate(lines):
            qa_dict = json.loads(line)
            instruction = qa_dict['instruction']
            answer = qa_dict['answer']
            if self.is_few_shot:
                examples = self.get_fewshot_examples(task_name)
                instruction = f"{instruction}\n{examples}"
            if task_name == "5_1" or task_name == "5_1_few_shot":
                input_text = qa_dict['input'] + '\n' + '摘要:'
            elif task_name == "5_2" or task_name == "5_2_few_shot":
                input_text = qa_dict['input'] + '\n' + '裁判分析过程:'
            elif task_name == "5_3" or task_name == "5_3_few_shot":
                input_text = qa_dict['input'] + '\n' + '翻译结果:'
            else:
                input_text = qa_dict['input'] + '\n' + '答案:'
            instruction_ls.append(instruction)
            input_text_ls.append(input_text)
            answer_ls.append(answer)
        return instruction_ls, input_text_ls, answer_ls