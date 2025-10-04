#!/bin/bash

export PYTHONPATH="path/to/project:$PYTHONPATH"

# Configuration - All paths are preset
UNTUNED_SCORES="path/to/eval_results/sd1-5/untuned/qwen_2-5-vl-7b-it_untuned_scores.json"
DPO_SCORES="path/to/eval_results/sd1-5/dpo-trained-semantic_relation_combined/qwen_2-5-vl-7b-it_dpo-trained-semantic_relation_combined_scores.json"
OUTPUT_DIR="path/to/analysis_results/dpo-trained-semantic_relation_combined"
METRICS=("semantic_coverage" "relation_validity" "style_score")

# Generate line plots for each metric
for metric in "${METRICS[@]}"; do
    python3 path/to/score_analysis.py \
        --score_files "$UNTUNED_SCORES" "$DPO_SCORES" \
        --labels "Untuned" "DPO-Trained" \
        --metric "$metric" \
        --plot_type "line" \
        --output "$OUTPUT_DIR/${metric}_line_plot.png" > /dev/null 2>&1
done

echo "Line plot analysis completed!"
echo "Charts saved to: $OUTPUT_DIR"

