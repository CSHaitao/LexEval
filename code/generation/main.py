import argparse
import os
import json
import logging
from Alpaca_gen import Alpaca_generator
from Qwen_gen import Qwen_generator
from Baichuan_base_gen import Baichuan_base_generator
from Baichuan_chat_gen import Baichuan_chat_generator
from Chatglm_gen import Chatglm_generator
from Internlm_base_gen import Internlm_base_generator
from Internlm_chat_gen import Internlm_chat_generator
from Llama_base_gen import Llama_base_generator
from Llama_chat_gen import Llama_chat_generator
from Chatgpt_gen import Chatgpt_generator
from Chinese_alpaca_gen import Chinese_alpaca_generator
from Tigerbot_gen import Tigerbot_generator
from Belle_llama_gen import Belle_llama_generator
from Fuzi_gen import Fuzi_generator
from Chatlaw_gen import Chatlaw_generator
from Xverse_gen import XVERSE_generator
from MPT_base_gen import MPT_base_generator
from MPT_instruct_gen import MPT_instruct_generator
from Chinese_llama_gen import Chinese_llama_generator
from Gogpt_gen import Gogpt_generator
from Ziya_gen import Ziya_generator
from Vicuna_gen import Vicuna_generator
from Mossmoon_gen import Mossmoon_generator
from Lexilaw_gen import Lexilaw_generator
from Lawyer_llama_gen import Lawyer_llama_generator
from Wisdom_gen import Wisdom_generator
from Lawgpt_gen import Lawgpt_generator
from Hanfei_gen import Hanfei_generator

def main(args):
    # Create a logger
    logging.basicConfig(filename=args.log_name, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    if args.model_name == 'Alpaca_7B':
        llm_generator = Alpaca_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Qwen_14B_chat' or args.model_name == 'Qwen_7B_chat':
        llm_generator = Qwen_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Baichuan_13B_base' or args.model_name == 'Baichuan_7B_base':
        llm_generator = Baichuan_base_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Baichuan_13B_chat':
        llm_generator = Baichuan_chat_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Chatglm_6B' or args.model_name == 'Chatglm2_6B' or args.model_name == 'Chatglm3_6B':
        llm_generator = Chatglm_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Internlm_7B':
        llm_generator = Internlm_base_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Internlm_7B_chat':
        llm_generator = Internlm_chat_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Llama_2_7B' or args.model_name == 'Chinese_llama_13B':
        llm_generator = Llama_base_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Llama_2_7B_chat' or args.model_name == 'Llama_2_13B_chat':
        llm_generator = Llama_chat_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Chinese_alpaca':
        llm_generator = Chinese_alpaca_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Tigerbot_base':
        llm_generator = Tigerbot_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Belle_llama':
        llm_generator = Belle_llama_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Fuzi':
        llm_generator = Fuzi_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Chatlaw_33B' or args.model_name == 'Chatlaw_13B':
        llm_generator = Chatlaw_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization, model_path_base=args.model_path_base)
    elif args.model_name == 'XVERSE':
        llm_generator = XVERSE_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'MPT_7B':
        llm_generator = MPT_base_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'MPT_7B_instruct':
        llm_generator = MPT_instruct_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Chinese_llama_7B':
        llm_generator = Chinese_llama_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Gogpt_7B' or args.model_name == 'Gogpt_13B':
        llm_generator = Gogpt_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Ziya_llama':
        llm_generator = Ziya_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Vicuna':
        llm_generator = Vicuna_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Mossmoon':
        llm_generator = Mossmoon_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Lawyer_llama':
        llm_generator = Lawyer_llama_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Wisdom':
        llm_generator = Wisdom_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Lawgpt_10' or args.model_name == 'Lawgpt_11':
        llm_generator = Lawgpt_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Hanfei':
        llm_generator = Hanfei_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'Lexilaw':
        llm_generator = Lexilaw_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, is_vllm=args.is_vllm, few_shot_path=args.few_shot_path, model_path=args.model_path, model_name=args.model_name, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.model_name == 'gpt-4' or args.model_name == 'gpt-3.5-turbo':
        llm_generator = Chatgpt_generator(f_path=args.f_path, is_few_shot=args.is_few_shot, device=args.device, few_shot_path=args.few_shot_path, api_base=args.api_base, api_key=args.api_key, model_name=args.model_name)
    
    if args.model_name != 'gpt-4' and args.model_name != 'gpt-3.5-turbo':    
        model, tokenizer = llm_generator.model_init()
    else:
        model, tokenizer = None, None
    task_num = args.f_path.split("/")[-1].split(".")[0].split("_")[0]
    q_type = "multiple_choice" if task_num != "5" else "generation"
    logging.info(f"Start running {args.model_name} on task {task_num}_{args.f_path.split('/')[-1].split('.')[0].split('_')[1]}")
    output_ls, num_unsuccess = llm_generator.generate_output(q_type=q_type, tokenizer=tokenizer, model=model, batch_size=args.batch_size)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    outfile_name = os.path.join(args.output_dir, f"{args.model_name}_{args.f_path.split('/')[-1]}l")
    outfile = open(outfile_name, 'w', encoding='utf8')
    for save_dict in output_ls:
        outline = json.dumps(save_dict,ensure_ascii=False)+'\n'
        outfile.write(outline)
    # Write number of failure generations into the log
    if num_unsuccess != 0:
        logging.info(f"Number of failure generations for {args.f_path}: {num_unsuccess}")
    outfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_path", type=str, default="/liuzyai04/thuir/cy/data/1_1.json")
    parser.add_argument("--is_few_shot", action="store_true")
    parser.add_argument("--is_vllm", action="store_true")
    parser.add_argument("--few_shot_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="/liuzyai04/thuir/cy/model/Alpaca_7B_v1")
    parser.add_argument("--model_path_base", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="Alpaca_7B")
    parser.add_argument("--output_dir", type=str, default="/liuzyai04/thuir/lht/test_legal/result/Alpaca_7B")
    parser.add_argument("--log_name", type=str, default='running.log')
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()
    main(args)
    