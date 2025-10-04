#!/bin/bash

export PYTHONPATH="path/to/project:$PYTHONPATH"
export WANDB_MODE=offline
# export NVTE_FRAMEWORK=pytorch
# export NVTE_DISABLE_TE=1

# Configuration: example metric is semantic_relation_combined
MODEL_NAME="path/to/pretrained-model/sd1-5"
DATASET_PATH="path/to/preference_datasets/sd1-5_semantic_relation_combined_hf"
OUTPUT_DIR="path/to/output/dpo-sd1-5-semantic_relation_combined"

NVTE_FRAMEWORK=pytorch NVTE_DISABLE_TE=1 accelerate launch \
  --num_processes=1 \
  --mixed_precision="fp16" \
  train.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --dataset_name="$DATASET_PATH" \
  --train_batch_size=1 \
  --dataloader_num_workers=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --learning_rate=2e-7 \
  --scale_lr \
  --checkpointing_steps=500 \
  --beta_dpo=5000 \
  --output_dir="$OUTPUT_DIR" \
  --mixed_precision="fp16" \
  --seed=42 \
  --report_to="tensorboard"
