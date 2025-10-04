#!/bin/bash

export NVTE_FRAMEWORK=pytorch
export NVTE_DISABLE_TE=1

python3 path/to/matrics_eval.py \
  --results_base "path/to/eval_results/sd_xl/untuned" \
  --results_tags "qwen_2-5-vl-7b-it_untuned"
