#!/bin/bash

export NVTE_FRAMEWORK=pytorch
export NVTE_DISABLE_TE=1

python3 path/to/extract.py \
  --prompts_file "path/to/prompts.txt" \
  --output_file "path/to/eval_results/templates.json"
