#!/bin/bash
cd generation

# Enter the model's name, model's path and device
MODEL_NAME_LS=("xxx" "yyy" "zzz")
MODELPATH_NAME_LS=("xxx_path" "yyy_path" "zzz_path") # You can specify your own model and its path
DATA_DIR="../../data" # The directory path for original data
EXAMPLE_DIR="sss" # The directory path for few-shot examples
DEVICE="0" # You can specify your own device

# Run zero-shot examples
for i in "${!MODEL_NAME_LS[@]}"
do
    for j in 1 2 3 4
    do
        python main.py \
            --f_path $DATA_DIR/1_$j.json \
            --model_path ${MODELPATH_NAME_LS[$i]} \
            --model_name ${MODEL_NAME_LS[$i]} \
            --device $DEVICE \
            --output_dir ../../model_output/zero_shot/${MODEL_NAME_LS[$i]} \
            --log_name "running.log"
    done
    for j in 1 2 3 4 5
    do
        python main.py \
            --f_path $DATA_DIR/2_$j.json \
            --model_path ${MODELPATH_NAME_LS[$i]} \
            --model_name ${MODEL_NAME_LS[$i]} \
            --device $DEVICE \
            --output_dir ../../model_output/zero_shot/${MODEL_NAME_LS[$i]} \
            --log_name "running.log"
    done
    for j in 1 2 3 4 5 6
    do
        python main.py \
            --f_path $DATA_DIR/3_$j.json \
            --model_path ${MODELPATH_NAME_LS[$i]} \
            --model_name ${MODEL_NAME_LS[$i]} \
            --device $DEVICE \
            --output_dir ../../model_output/zero_shot/${MODEL_NAME_LS[$i]} \
            --log_name "running.log"
    done
    for j in 1 2
    do
        python main.py \
            --f_path $DATA_DIR/4_$j.json \
            --model_path ${MODELPATH_NAME_LS[$i]} \
            --model_name ${MODEL_NAME_LS[$i]} \
            --device $DEVICE \
            --output_dir ../../model_output/zero_shot/${MODEL_NAME_LS[$i]} \
            --log_name "running.log"
    done
    for j in 1 2 3 4
    do
        python main.py \
            --f_path $DATA_DIR/5_$j.json \
            --model_path ${MODELPATH_NAME_LS[$i]} \
            --model_name ${MODEL_NAME_LS[$i]} \
            --device $DEVICE \
            --output_dir ../../model_output/zero_shot/${MODEL_NAME_LS[$i]} \
            --log_name "running.log"
    done
    for j in 1 2 3
    do
        python main.py \
            --f_path $DATA_DIR/6_$j.json \
            --model_path ${MODELPATH_NAME_LS[$i]} \
            --model_name ${MODEL_NAME_LS[$i]} \
            --device $DEVICE \
            --output_dir ../../model_output/zero_shot/${MODEL_NAME_LS[$i]} \
            --log_name "running.log"
    done
done

# Run few-shot examples
for i in "${!MODEL_NAME_LS[@]}"
do
    for j in 1 2 3 4
    do
        python main.py \
            --f_path $DATA_DIR/1_$j.json \
            --few_shot_path $EXAMPLE_DIR/1_${j}_few_shot.json \
            --model_path ${MODELPATH_NAME_LS[$i]} \
            --model_name ${MODEL_NAME_LS[$i]} \
            --device $DEVICE \
            --output_dir ../../model_output/few_shot/${MODEL_NAME_LS[$i]} \
            --log_name "running.log" \
            --is_few_shot
    done
    for j in 1 2 3 4 5
    do
        python main.py \
            --f_path $DATA_DIR/2_$j.json \
            --few_shot_path $EXAMPLE_DIR/2_${j}_few_shot.json \
            --model_path ${MODELPATH_NAME_LS[$i]} \
            --model_name ${MODEL_NAME_LS[$i]} \
            --device $DEVICE \
            --output_dir ../../model_output/few_shot/${MODEL_NAME_LS[$i]} \
            --log_name "running.log" \
            --is_few_shot
    done
    for j in 1 2 3 4 5 6
    do
        python main.py \
            --f_path $DATA_DIR/3_$j.json \
            --few_shot_path $EXAMPLE_DIR/3_${j}_few_shot.json \
            --model_path ${MODELPATH_NAME_LS[$i]} \
            --model_name ${MODEL_NAME_LS[$i]} \
            --device $DEVICE \
            --output_dir ../../model_output/few_shot/${MODEL_NAME_LS[$i]} \
            --log_name "running.log" \
            --is_few_shot
    done
    for j in 1 2
    do
        python main.py \
            --f_path $DATA_DIR/4_$j.json \
            --few_shot_path $EXAMPLE_DIR/4_${j}_few_shot.json \
            --model_path ${MODELPATH_NAME_LS[$i]} \
            --model_name ${MODEL_NAME_LS[$i]} \
            --device $DEVICE \
            --output_dir ../../model_output/few_shot/${MODEL_NAME_LS[$i]} \
            --log_name "running.log" \
            --is_few_shot
    done
    for j in 1 2 3 4
    do
        python main.py \
            --f_path $DATA_DIR/5_$j.json \
            --few_shot_path $EXAMPLE_DIR/5_${j}_few_shot.json \
            --model_path ${MODELPATH_NAME_LS[$i]} \
            --model_name ${MODEL_NAME_LS[$i]} \
            --device $DEVICE \
            --output_dir ../../model_output/few_shot/${MODEL_NAME_LS[$i]} \
            --log_name "running.log" \
            --is_few_shot
    done
    for j in 1 2 3
    do
        python main.py \
            --f_path $DATA_DIR/6_$j.json \
            --few_shot_path $EXAMPLE_DIR/6_${j}_few_shot.json \
            --model_path ${MODELPATH_NAME_LS[$i]} \
            --model_name ${MODEL_NAME_LS[$i]} \
            --device $DEVICE \
            --output_dir ../../model_output/few_shot/${MODEL_NAME_LS[$i]} \
            --log_name "running.log" \
            --is_few_shot
    done
done