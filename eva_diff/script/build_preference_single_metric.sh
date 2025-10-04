#!/bin/bash

export PYTHONPATH="path/to/project:$PYTHONPATH"

# Configuration
SCORES_FILE="path/to/eval_results/sd1-5/untuned/qwen_2-5-vl-7b-it_untuned_scores.json"
PROMPTS_FILE="path/to/prompts.txt"
IMAGES_DIR="path/to/image_output/sd1-5/untuned"
BASE_OUTPUT_DIR="path/to/preference_datasets"

# Change this to: "semantic_coverage", "relation_validity", or "style_score"
# PRIMARY_METRIC="semantic_coverage"
PRIMARY_METRIC="relation_validity"

OUTPUT_PKL="$BASE_OUTPUT_DIR/sd1-5_${PRIMARY_METRIC}.pkl"
OUTPUT_HF="$BASE_OUTPUT_DIR/sd1-5_${PRIMARY_METRIC}_hf"

echo "Building preference dataset with metric: $PRIMARY_METRIC"

mkdir -p "$(dirname "$OUTPUT_PKL")"

# Build PKL dataset
python3 eva_diff/build_preference_dataset.py \
  --scores_file "$SCORES_FILE" \
  --prompts_file "$PROMPTS_FILE" \
  --images_dir "$IMAGES_DIR" \
  --output_file "$OUTPUT_PKL" \
  --primary_metric "$PRIMARY_METRIC"

if [[ ! -f "$OUTPUT_PKL" ]]; then
    echo "Error: Failed to generate PKL file"
    exit 1
fi

# Convert to HuggingFace format
python3 eva_diff/preference_dataset_loader.py \
  --dataset_path "$OUTPUT_PKL" \
  --convert_to_hf "$OUTPUT_HF"

if [[ ! -d "$OUTPUT_HF" ]]; then
    echo "Error: Failed to convert to HuggingFace format"
    exit 1
fi

echo "Success! Dataset saved to: $OUTPUT_HF"
echo "For training use: --dataset_name=\"$OUTPUT_HF\""
