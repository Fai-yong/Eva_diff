#!/bin/bash

export NVTE_FRAMEWORK=pytorch
export NVTE_DISABLE_TE=1

# only enable cuda 0
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="path/to/pretrained_models/qwen_2-5-vl-7b-it"
IMAGES_DIR="path/to/image_output/sd1-5/dpo-trained-relation_validity"
SAVE_DIR="path/to/eval_results/sd1-5/dpo-trained-relation_validity"
FILE_NAME="qwen_2-5-vl-7b-it_dpo-trained-relation_validity_eval_results.json"
TEMPLATE_FILE="path/to/eval_results/templates.json"

python path/to/run_eval.py \
  --model_type qwen2-5-vl \
  --model_path $MODEL_PATH \
  --images_dir $IMAGES_DIR \
  --template_file $TEMPLATE_FILE \
  --save_dir $SAVE_DIR \
  --file_name $FILE_NAME \
  --device cuda:0

