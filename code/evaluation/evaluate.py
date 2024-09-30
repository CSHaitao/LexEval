import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jsonlines
import torch
import torch.nn as nn
import pandas as pd
import bert_score
import argparse
import jieba
from transformers import BertTokenizer, BartForConditionalGeneration
from typing import List
from rouge import Rouge
from process import BARTScorer, find_valid_substrings, normalize_zh_answer
from tqdm import tqdm

import numpy as np

class Evaluator:
    '''
    Evaluating the score for a given task on a given metric from one model's output
    '''
    def __init__(self, file_path, task_type, metric, device='cpu', model_path=None):
        '''
        Args:
            file_path: Input file path for the model's output
            task_type: generation or multiple_choice task
            metric: metrics for evaluation, f1 or accuracy for multiple choice and rouge-l, bertscore or bartscore for generation task
            device: Using cuda or cpu to do evaluation
            model_path: path for bert model or bart model, only useful if using bertscore or bartscore to evaluate
        '''
        self.file_path = file_path
        if task_type == 'generation':
            self.task_type = task_type
            if metric == 'Rouge_L':
                self.metric = metric
            elif metric == 'Bertscore':
                self.metric = metric
                if model_path != None:
                    self.model_path = model_path
                else:
                    raise ValueError(f"Lacking bert model for evaluation")
            elif metric == 'Bartscore':
                self.metric = metric
                if model_path != None:
                    self.model_path = model_path
                else:
                    raise ValueError(f"Lacking bart model for evaluation")
            else:
                raise ValueError(f"Wrong metric for generation evaluation. It has to be 'Rouge_L', 'Bertscore' or 'Bartscore' but get {metric}")
        elif task_type == 'multiple_choice':
            self.task_type = task_type
            if metric == 'Accuracy' or metric == 'F1':
                self.metric = metric
            else:
                raise ValueError(f"Wrong metric for multiple choice evaluation. It has to be 'Accuracy' or 'F1' but get {metric}")
        else:
            raise ValueError(f"Wrong task type for evaluation. It has to be 'generation' or 'multiple_choice', but get {task_type}")
        self.device = device
        
    def eval_accuracy(self):
        '''
        Output the accuracy for the given file
        '''
        score = 0
        num = 0
        with jsonlines.open(self.file_path) as f:
            for qa_one in f:
                pred = find_valid_substrings(qa_one['output'])
                if pred == qa_one['answer']:
                    score += 1
                num += 1
        acc = score / num
        return acc
    
    def eval_f1(self):
        '''
        Output the f1-score for the given file, refers to lawbench
        '''    
        with jsonlines.open(self.file_path) as f:
            score = []
            for qa_one in f:
                pred = find_valid_substrings(qa_one['output'])
                pred_set = set(pred)
                gt_set = set(qa_one['answer'])
                precision = len(pred_set.intersection(gt_set)) / len(pred_set) if len(pred_set) > 0 else 0
                recall = len(pred_set.intersection(gt_set)) / len(gt_set) if len(gt_set) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                score.append(f1)
            f1 = sum(score) / len(score)   
        return f1
    
    def eval_rougel(self):
        '''
        Output the Rouge-L score for the given file
        '''
        with jsonlines.open(self.file_path) as f:
            score = []
            for qa_one in f:
                pred = " ".join(list(jieba.cut(normalize_zh_answer(qa_one['output']), cut_all=False)))
                ans = " ".join(list(jieba.cut(normalize_zh_answer(qa_one['answer']), cut_all=False)))
                rouge = Rouge()
                try:
                    score.append(rouge.get_scores([pred], [ans], avg=True)["rouge-l"]["f"])
                except:
                    score.append(0.0)
        rouge_l = sum(score) / len(score)
        return rouge_l
    
    def eval_bertscore(self, batch_size=10):
        '''
        Output the bertscore for the given file
        '''
        with jsonlines.open(self.file_path) as f:
            all_qa = [qa_one for qa_one in f]
        all_pred, all_gt = [qa_one['output'] for qa_one in all_qa], [qa_one['answer'] for qa_one in all_qa]
        assert (len(all_pred) == len(all_gt))
        score_p, score_r, score_f1 = bert_score.score(all_pred, all_gt, lang='zh', verbose=False, model_type=self.model_path, num_layers=8, device=self.device, batch_size=batch_size)
        bertscore = (sum(score_f1) / len(score_f1)).item()
        return bertscore
    
    def eval_bartscore(self, batch_size=10):
        '''
        Output the bartscore for the given file
        '''
        with jsonlines.open(self.file_path) as f:
            all_qa = [qa_one for qa_one in f]
        all_pred, all_gt = [qa_one['output'] for qa_one in all_qa], [qa_one['answer'] for qa_one in all_qa]
        bart_calculator = BARTScorer(checkpoint=self.model_path)
        score = bart_calculator.score(all_pred, all_gt, batch_size=batch_size)
        bartscore = sum(score) / len(score)
        return bartscore
    
    def eval(self):
        '''
        Output the evaluation result for the given file on the given metric
        '''
        if self.task_type == 'generation':
            if self.metric == 'Rouge_L':
                return self.eval_rougel()
            elif self.metric == 'Bertscore':
                return self.eval_bertscore()
            elif self.metric == 'Bartscore':
                return self.eval_bartscore()
        elif self.task_type == 'multiple_choice':
            if self.metric == 'Accuracy':
                return self.eval_accuracy()
            elif self.metric == 'F1':
                return self.eval_f1()
        
def main(input_dir, output_dir, metrics_choice, metrics_gen, device='cpu', model_path=None):
    '''
    Given the input directory for the model's generation and metrics, output the correspondent score
    '''
    all_model = os.listdir(input_dir)
    # Only select directory
    all_model = [all_model[i] for i in range(len(all_model)) if os.path.isdir(input_dir + '/' + all_model[i])]
    results = {"task": [], "model": [], "metrics": [], "score": []}
    for model in all_model:
        print(f"Begin evaluating on model {model}")
        model_dir = os.path.join(input_dir, model)
        all_file = os.listdir(model_dir)
        # Only retain .jsonl files, which store the output of models and gold answers
        all_file = [all_file[i] for i in range(len(all_file)) if all_file[i].endswith(".jsonl")]
        for i in tqdm(range(len(all_file))):
            f_name = all_file[i]
            # Extract the task name
            task_name = f_name.split('_')[-2] + '_' + f_name.split('_')[-1].split('.')[0]
            assert int(task_name.split('_')[0]) <= 6 and int(task_name.split('_')[0]) >= 1 and int(task_name.split('_')[1]) >= 1 and int(task_name.split('_')[1]) <= 6, f"Wrong task type, the task type needs to follow 'i_j' format, where 1 <= i, j <= 6 but get {task_name}"
            results['task'].append(task_name)
            results['model'].append(model)
            if task_name.split('_')[0] == '5':
                results['metrics'].append(metrics_gen)
                evaluator = Evaluator(file_path=os.path.join(model_dir, f_name), task_type='generation', metric=metrics_gen, device=device, model_path=model_path)
                results['score'].append(evaluator.eval())
            else:
                results['metrics'].append(metrics_choice)
                evaluator = Evaluator(file_path=os.path.join(model_dir, f_name), task_type='multiple_choice', metric=metrics_choice)
                results['score'].append(evaluator.eval())
    results = pd.DataFrame(results)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)               
    results.to_csv(os.path.join(output_dir, 'evaluation_result.csv'))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None, help="The directory storing all the model's generation")
    parser.add_argument("--output_dir", type=str, default=None, help="The directory for saving output score")
    parser.add_argument("--metrics_choice", type=str, default="Accuracy", help="metrics for multiple choice questions, can be 'Accuracy' or 'F1'")
    parser.add_argument("--metrics_gen", type=str, default="Rouge_L", help="metrics for generation tasks, can be 'Rouge_L', 'Bertscore' or 'Bartscore'")
    parser.add_argument("--model_path", type=str, default=None, help="file path for bert model or bart model")
    parser.add_argument("--device",type=str, default="cpu", help="device for evaluation, can be cpu or cuda")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir, metrics_choice=args.metrics_choice, metrics_gen=args.metrics_gen, device=args.device, model_path=args.model_path)