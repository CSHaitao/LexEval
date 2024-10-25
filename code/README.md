# Usage of evaluation code
The evaluation process mainly consists of two steps: "model result generation" and "model result evaluation".
## Model Result Generation
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