#!/bin/bash

# 使用DPO训练后的模型生成图像

export PYTHONPATH="path/to/project:$PYTHONPATH"

# Configuration
DPO_MODEL_PATH="path/to/output/dpo-sd1-5-semantic_relation_combined"
PROMPTS_FILE="path/to/prompts.txt"
OUTPUT_DIR="path/to/image_output/sd1-5/dpo-trained-semantic_relation_combined"

python3 path/to/sd.py \
  --prompts_file "$PROMPTS_FILE" \
  --save_path "$OUTPUT_DIR" \
  --model_type untuned \
  --model_path "$DPO_MODEL_PATH" \
  --device cuda:0 \
  --inference_steps 50 \
  --guidance_scale 7.5 \
  --num_images_per_prompt 4 \
  --seed 42
