#!/bin/bash

export NVTE_FRAMEWORK=pytorch
export NVTE_DISABLE_TE=1

python path/to/run_eval.py \
  --model_type llava \
  --model_path "path/to/pretrained-model/llava-onevision-7b" \
  --images_dir "path/to/image_output/sd_xl/untuned" \
  --template_file "path/to/eval_results/templates.json" \
  --output_file "path/to/eval_results/llava-1.6-7b_untuned_eval_results.json" \
  --device cuda:0 \
  --regenerate
