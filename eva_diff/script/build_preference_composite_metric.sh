#!/bin/bash

export PYTHONPATH="path/to/project:$PYTHONPATH"

# Configuration
SCORES_FILE="path/to/eval_results/sd1-5/untuned/qwen_2-5-vl-7b-it_untuned_scores.json"
PROMPTS_FILE="path/to/prompts.txt"
IMAGES_DIR="path/to/image_output/sd1-5/untuned"
BASE_OUTPUT_DIR="path/to/preference_datasets"

echo "Building composite metric preference datasets..."

mkdir -p "$BASE_OUTPUT_DIR"

# Build datasets for each composite metric
METRICS=(
    "semantic_coverage+relation_validity:semantic_relation_combined"
    "semantic_coverage+style_score:semantic_style_combined"
    "relation_validity+style_score:relation_style_combined"
    "semantic_coverage+relation_validity+style_score:all_metrics_combined"
)

for metric_pair in "${METRICS[@]}"; do
    metric_config="${metric_pair%%:*}"
    dataset_name="${metric_pair##*:}"

    output_pkl="$BASE_OUTPUT_DIR/sd1-5_${dataset_name}.pkl"
    output_hf="$BASE_OUTPUT_DIR/sd1-5_${dataset_name}_hf"

    echo "Building: $metric_config"

    python3 eva_diff/build_preference_dataset.py \
      --scores_file "$SCORES_FILE" \
      --prompts_file "$PROMPTS_FILE" \
      --images_dir "$IMAGES_DIR" \
      --output_file "$output_pkl" \
      --primary_metric "$metric_config"

    if [[ ! -f "$output_pkl" ]]; then
        echo "Error: Failed to generate $metric_config"
        continue
    fi

    python3 eva_diff/preference_dataset_loader.py \
      --dataset_path "$output_pkl" \
      --convert_to_hf "$output_hf"

    if [[ -d "$output_hf" ]]; then
        echo "Success: $output_hf"
    else
        echo "Error: Failed to convert $metric_config"
    fi
done

echo ""
echo "All datasets built. Use any of these for training:"
# echo "  --dataset_name=\"$BASE_OUTPUT_DIR/sd1-5_semantic_relation_combined_hf\""
# echo "  --dataset_name=\"$BASE_OUTPUT_DIR/sd1-5_semantic_style_combined_hf\""
# echo "  --dataset_name=\"$BASE_OUTPUT_DIR/sd1-5_relation_style_combined_hf\""
# echo "  --dataset_name=\"$BASE_OUTPUT_DIR/sd1-5_all_metrics_combined_hf\""
