<div align="center">
<img src="./figure/logo.jpg" style="width: 30%;height: 30%">
<h1> LexEval: A Comprehensive Chinese Legal Benchmark for Evaluating Large Language Models </h1>
</div>

<div align="center">

</div>

<div align="center">
  <!-- <a href="#model">Model</a> • -->
  🏆 <a href="https://collam.megatechai.com/">Leaderboard</a> |
  📚 <a href="https://huggingface.co/datasets/CSHaitao/LexEval">Data</a> |
  📃 <a href="https://arxiv.org/abs/2409.20288">Paper</a> |
  📊 <a href="#citation">Citation</a>
</div>

<p align="center">
    📖 <a href="README_ZH.md">   中文</a> | <a href="README.md">English</a>
</p>


Welcome to **LexEval**, the comprehensive Chinese legal benchmark designed to evaluate the performance of large language models (LLMs) in legal applications.




## What's New
- **[2024.10]** 🥳 LexEval is accepted by NeurIPs 2024

## Introduction

Large language models (LLMs) have made significant progress in natural language processing tasks and have shown considerable potential in the legal domain.  However, the legal applications often have high requirements on accuracy, reliability and fairness. Applying existing LLMs to legal systems without careful evaluation of their potentials and limitations could lead to significant risks in legal practice.
Therefore, to facilitate the healthy development and application of LLMs in the legal domain, we propose a comprehensive benchmark LexEval for evaluating LLMs in legal domain. 

Key aspects of LexEval:

**Ability Modeling:** We propose a novel Legal Cognitive Ability Taxonomy (LexAbility Taxonomy) to organize different legal tasks systematically. This taxonomy includes six core abilities: Memorization, Understanding, Logic Inference, Discrimination, Generation, and Ethics.

**Scale:** LexEval is currently the largest legal benchmark in China, comprising 23 tasks and 14,150 questions. Additionally, LexEval will be continuously updated to enable more comprehensive evaluations.

**Data Sources:** LexEval combines data from existing legal datasets, real-world exam datasets, and newly annotated datasets created by legal experts to provide a comprehensive assessment of LLM capabilities.

## Legal Cognitive Ability Taxonomy (LexCog)

Inspired by Bloom's taxonomy and real-world legal application scenarios, we propose a legal cognitive ability taxonomy (LexCog) to provide guidance for the evaluation of LLMs. Our taxonomy categorizes the application of LLMs in the legal domain into six ability levels: Memorization, Understanding, Logic Inference, Discrimination, Generation, and Ethic. 

<div align="center">

<img src="./figure/taxonomy.png">
<!-- <h1> A nice pic from our website </h1> -->

</div>

- **Memorization**: At this level, LLMs are responsible for recalling and accurately retrieving key legal information, such as fundamental statutes, case laws, legal principles, and specialized terminology. This ability ensures that models can store and access foundational knowledge relevant to legal tasks.

- **Understanding**: LLMs are tested on their ability to comprehend and interpret legal texts and concepts. They must grasp the meaning, implications, and relevance of legal content, demonstrating an ability to accurately understand the issues presented in legal cases, documents, or regulations.

- **Logic Inference**: This ability evaluates the model’s capacity for legal reasoning and deduction. LLMs are expected to apply logical reasoning to derive conclusions from given legal facts and rules. This includes identifying patterns, drawing inferences, and making connections between legal principles in a structured manner.

- **Discrimination**: LLMs must exhibit the ability to analyze, compare, and evaluate the significance of legal information based on specific legal criteria. This involves distinguishing between similar legal concepts, analyzing case precedents, and determining the relevance of legal evidence.

- **Generation**: At this level, LLMs are expected to produce professional and legally sound documents. This includes drafting legal opinions, contracts, case summaries, and other legal texts. The generated content should be precise, well-structured, and adhere to legal standards based on provided instructions or conditions.

- **Ethics**: The Ethics level assesses the model's capacity to recognize and address ethical issues in legal contexts. LLMs should be able to analyze legal ethical dilemmas, weigh pros and cons, and make decisions aligned with professional ethics, legal principles, and social values.

This taxonomy serves as the foundation of LexEval, ensuring that LLMs are evaluated comprehensively across the full spectrum of legal cognitive tasks. By organizing tasks into these six levels, the LexEval benchmark provides a robust framework for assessing how well LLMs can support legal professionals in their work, while also highlighting potential areas for improvement in the models' abilities to handle complex legal scenarios.


## Tasks Definition

The dataset for Lexeval consists of 14,150 questions carefully designed to cover the breadth of legal cognitive abilities outlined in the LexCog. The questions span 23 tasks relevant to legal scenarios, providing a diverse set for evaluating LLM performance.

The following table shows the details of the tasks in LexEval:
![image](./figure/tasks.png)

Further experimental details and analyses can be found in our paper.


## 🚀 Quick Start 

The evaluation process mainly consists of two steps: "model result generation" and "model result evaluation".

### Model Result Generation

* Prepare the data files, naming the original data files as `i_j.json`, and the few-shot examples files as `i_j_few_shot.json`. Here, `i_j` should be a task name composed of Arabic numerals and underlines.

* Directly run the `./generation/main.py`. Here is an example for `1_1` task:
    ```bash
    cd generation
    MODEL_PATH='xxx'
    MODEL_NAME='xxx'
    DATA_DIR='xxx'
    EXAMPLE_DIR='xxx'
    # Zero-shot
    python main.py \
        --f_path $DATA_DIR/1_1.json \
        --model_path $MODEL_PATH \
        --model_name $MODEL_NAME \
        --output_dir ../../model_output/zero_shot/$MODEL_NAME \
        --log_name running.log \
        --device "0"
    # Few-shot
    python main.py \
        --f_path $DATA_DIR/1_1.json \
        --few_shot_path $EXAMPLE_DIR/1_1_few_shot.json \
        --model_path $MODEL_PATH \
        --model_name $MODEL_NAME \
        --output_dir ../../model_output/few_shot/$MODEL_NAME \
        --log_name running.log \
        --device "0" \
        --is_few_shot
    # For some models, using vllm to make fast inference
    python main.py \
        --f_path $DATA_DIR/1_1.json \
        --model_path $MODEL_PATH \
        --model_name $MODEL_NAME \
        --output_dir ../../model_output/zero_shot/$MODEL_NAME \
        --log_name running.log \
        --device "0" \
        --batch_size 50 \
        --is_vllm
    ```

    * `--f_path`: The path for original data `i_j.json`.
    * `--model_path`: The path for model's checkpoint.
    * `--model_name`: The name of the model. You can find all available model names in `model_name.txt`.
    * `--output_dir`: The name of the output directory for model's generation.
    * `--log_name`: The path for the log.
    * `--is_few_shot`: Whether to use few-shot examples.
    * `--few_shot_path`: The path for few-shot examples, only valid in few-shot setting.
    * `--api_key`: The name of model's api key, only valid when using api.
    * `--api_base`: The name of model's api base, only valid when using api.
    * `--device`: The device id for `cuda`.
    * `--is_vllm`: Whether to use vllm for faster inference, and currently it is not available for all models.
    * `--batch_size`: The number of questions processed per inference time by vllm, only effective when using vllm.

* If you want to run multiple model results in batch, please refer to `run.sh`.
* Take zero-shot setting as an example. You can check the results in `./zero_shot_output/$MODEL_NAME`. The `.jsonl` file format for each line is as follows:
    ```python
    {"input": xxx, "output": xxx, "answer": xxx}
    ```

## Model Result Evaluation
* Directly run `./evaluation/evaluate.py`.
    ```bash
    cd evaluation
    python evaluate.py \
        --input_dir ../../model_output/zero_shot \
        --output_dir ../../evaluation_output \
        --metrics_choice "Accuracy" \
        --metrics_gen "Rouge_L" \
    ```
    * `--input_dir`: The directory path for the model's generation.
    * `--output_dir`: The output directory for evaluation result.
    * `--metrics_choice`: Evaluation metric for multiple-choice tasks, 'Accuracy' and 'F1' are currently supported.
    * `--metrics_gen`: Evaluation metric for generation tasks, 'Rouge_L', 'Bertscore' and 'Bartscore' are currently supported.
    * `--model_path`: The path for bert and bart model, only valid when the evaluation metric is 'Bertscore' or 'Bartscore'.
    * `--device`: The device id for `cuda`.

* Go to `./evaluation_output/evaluation_result.csv` to check the full results. The example format is as follows:
    |task|model|metrics|score|
    |:--:|:---:|:-----:|:---:|
    |1_1|Chatglm3_6B|Accuracy|0.192|
    |5_1|Baichuan_13B_base|Rouge_L|0.215|
    |...|...|...|...|





## Contributing

LexEval is an ongoing project, and we welcome contributions from the community. You can contribute by:

* Adding new tasks to the LexAbility Taxonomy

* Submitting new datasets or annotations

* Improving the evaluation framework

Please contact liht22@mails.tsinghua.edu.cn

## License

LexEval is released under the [MIT License](LICENSE).


## Citation
If you find our work useful, please do not save your star and cite our work:

```bibtex
@misc{li2024lexevalcomprehensivechineselegal,
      title={LexEval: A Comprehensive Chinese Legal Benchmark for Evaluating Large Language Models}, 
      author={Haitao Li and You Chen and Qingyao Ai and Yueyue Wu and Ruizhe Zhang and Yiqun Liu},
      year={2024},
      eprint={2409.20288},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.20288}, 
}
```
