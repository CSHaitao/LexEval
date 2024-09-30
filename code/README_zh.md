# 评测代码使用方法
模型评测主要分为“模型结果生成”与“模型结果评估”两步。
## 模型结果生成
* 准备数据文件，对于原始数据以`i_j.json`命名，对于few-shot examples文件，应以`i_j_few_shot.json`命名，其中`i_j`应为由阿拉伯数字和下划线组成的任务名。
* 直接运行`./generation/main.py`文件。以`1_1`任务为例。
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
        --output_dir ../zero_shot_output/$MODEL_NAME \
        --log_name running.log \
        --device "0"
    # Few-shot
    python main.py \
        --f_path $DATA_DIR/1_1.json \
        --few_shot_path $EXAMPLE_DIR/1_1_few_shot.json \
        --model_path $MODEL_PATH \
        --model_name $MODEL_NAME \
        --output_dir ../few_shot_output/$MODEL_NAME \
        --log_name running.log \
        --device "0" \
        --is_few_shot
    # For some models, using vllm to make fast inference
    python main.py \
        --f_path $DATA_DIR/1_1.json \
        --model_path $MODEL_PATH \
        --model_name $MODEL_NAME \
        --output_dir ../zero_shot_output/$MODEL_NAME \
        --log_name running.log \
        --device "0" \
        --batch_size 50 \
        --is_vllm
    ```
    * `--f_path`: 原始数据`i_j.json`所在路径
    * `--model_path`: 模型权重所在路径
    * `--model_name`: 模型名称，可在`model_name.txt`查看可用模型名
    * `--output_dir`: 输出文件夹名称
    * `--log_name`: 日志所在路径
    * `--is_few_shot`: 是否使用few-shot examples
    * `--few_shot_path`: few-shot examples路径，仅在使用few-shot examples时有效
    * `--api_key`: 模型api key的名称，仅在使用api时有效
    * `--api_base`: 模型api base名称，仅在使用api时有效
    * `--device`: 运行使用`cuda`编号
    * `--is_vllm`: 是否使用vllm进行快速推理，目前支持部分模型
    * `--batch_size`: vllm一次推理的问题数，仅在使用vllm时有效
* 如果想批量运行多个模型结果，请参考脚本`run.sh`。
* 以zero_shot为例，运行完毕后可以进入`./zero_shot_output/$MODEL_NAME`文件夹查看结果`.jsonl`文件，每一行格式如下：
    ```python
    {"input": xxx, "output": xxx, "answer": xxx}
    ```
## 模型结果评估
* 直接运行`./evaluation/evaluate.py`文件
    ```bash
    cd evaluation
    python evaluate.py \
        --input_dir ../zero_shot_output \
        --output_dir ../evaluation_output \
        --metrics_choice "Accuracy" \
        --metrics_gen "Rouge_L" \
    ```
    * `--input_dir`: 所有要评估模型生成结果的文件夹
    * `--output_dir`: 评测结果输出文件夹
    * `--metrics_choice`: 选择题任务评测指标，目前支持'Accuracy'与'F1'
    * `--metrics_gen`: 内容生成任务评测指标，目前支持'Rouge_L', 'Bertscore'和'Bartscore'
    * `--model_path`: bert或bart模型所在路径，仅当生成任务指标选择为'Bertscore'或'Bartscore'时有效
    * `--device`: 运行cuda编号
* 进入`./evaluation_output/evaluation_result.csv`查看运行完毕的结果，示例格式如下 
    |task|model|metrics|score|
    |:--:|:---:|:-----:|:---:|
    |1_1|Chatglm3_6B|Accuracy|0.192|
    |5_1|Baichuan_13B_base|Rouge_L|0.215|
    |...|...|...|...|


