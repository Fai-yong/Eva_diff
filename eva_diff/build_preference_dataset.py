#!/usr/bin/env python3
"""
Build preference dataset from evaluation results for DPO training.

This script reads evaluation scores and creates preference pairs based on:
1. Primary metric (default: semantic_coverage)
2. Secondary tie-breaker: object_num (prefer fewer objects)
3. Tertiary tie-breaker: filename suffix (prefer smaller suffix)
"""

import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import io
from PIL import Image


def extract_prompt_index_and_suffix(filename: str) -> Tuple[int, int]:
    """
    Extract prompt index and image suffix from filename.

    Args:
        filename: e.g., "0_1.png" or "5.png"

    Returns:
        (prompt_index, suffix) e.g., (0, 1) or (5, 0)
    """
    name_without_ext = filename.split('.')[0]
    if '_' in name_without_ext:
        parts = name_without_ext.split('_')
        return int(parts[0]), int(parts[1])
    else:
        return int(name_without_ext), 0


def load_evaluation_results(scores_file: str) -> Dict[str, Dict]:
    """Load evaluation scores from JSON file."""
    with open(scores_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompts(prompts_file: str) -> List[str]:
    """Load prompts from text file."""
    with open(prompts_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


def group_images_by_prompt_index(scores: Dict[str, Dict]) -> Dict[int, List[Tuple[str, Dict]]]:
    """Group images by their prompt index."""
    groups = defaultdict(list)

    for filename, score_data in scores.items():
        if score_data is None:  # Skip failed evaluations
            continue

        prompt_index, suffix = extract_prompt_index_and_suffix(filename)
        groups[prompt_index].append((filename, score_data))

    return dict(groups)


def calculate_composite_score(score_data: Dict, metric_config: str) -> float:
    """
    Calculate composite score based on metric configuration.

    Args:
        score_data: Dictionary containing evaluation scores
        metric_config: Metric configuration string

    Returns:
        Calculated composite score
    """
    if metric_config in ["semantic_coverage", "relation_validity", "style_score"]:
        # Single metric
        return score_data.get(metric_config, 0)
    elif metric_config == "semantic_coverage+relation_validity":
        # Average of semantic_coverage and relation_validity
        semantic = score_data.get("semantic_coverage", 0)
        relation = score_data.get("relation_validity", 0)
        return (semantic + relation) / 2.0
    elif metric_config == "semantic_coverage+style_score":
        # Average of semantic_coverage and style_score
        semantic = score_data.get("semantic_coverage", 0)
        style = score_data.get("style_score", 0)
        return (semantic + style) / 2.0
    elif metric_config == "relation_validity+style_score":
        # Average of relation_validity and style_score
        relation = score_data.get("relation_validity", 0)
        style = score_data.get("style_score", 0)
        return (relation + style) / 2.0
    elif metric_config == "semantic_coverage+relation_validity+style_score":
        # Average of all three metrics
        semantic = score_data.get("semantic_coverage", 0)
        relation = score_data.get("relation_validity", 0)
        style = score_data.get("style_score", 0)
        return (semantic + relation + style) / 3.0
    else:
        raise ValueError(f"Unsupported metric configuration: {metric_config}")


def select_preference_pair(
    images: List[Tuple[str, Dict]],
    primary_metric: str = "semantic_coverage"
) -> Optional[Tuple[Tuple[str, Dict], Tuple[str, Dict]]]:
    """
    Select the best and worst images from a group based on scoring criteria.

    Args:
        images: List of (filename, score_data) tuples
        primary_metric: Primary metric to use for selection (supports composite metrics)

    Returns:
        ((best_filename, best_scores), (worst_filename, worst_scores)) or None
    """
    if len(images) < 2:
        return None

    def get_sort_key(item):
        filename, score_data = item
        # Calculate primary score using composite score function
        primary_score = calculate_composite_score(score_data, primary_metric)
        object_num = score_data.get("object_num", 0)
        _, suffix = extract_prompt_index_and_suffix(filename)

        # Return tuple for sorting: (primary_score, -object_num, -suffix)
        # Negative values because we want ascending order for object_num and suffix
        return (primary_score, -object_num, -suffix)

    # Sort by criteria: highest primary score, fewer objects, smaller suffix
    sorted_images = sorted(images, key=get_sort_key, reverse=True)

    best = sorted_images[0]   # Highest score
    worst = sorted_images[-1]  # Lowest score

    # Check if they're actually different
    best_key = get_sort_key(best)
    worst_key = get_sort_key(worst)

    if best_key == worst_key:
        return None  # All images have the same scores

    return best, worst


def image_to_bytes(image_path: str) -> bytes:
    """Convert image file to bytes."""
    with open(image_path, 'rb') as f:
        return f.read()


def build_preference_dataset(
    scores_file: str,
    prompts_file: str,
    images_dir: str,
    output_file: str,
    primary_metric: str = "semantic_coverage"
):
    """
    Build preference dataset from evaluation results.

    Args:
        scores_file: Path to evaluation scores JSON file
        prompts_file: Path to prompts text file
        images_dir: Directory containing generated images
        output_file: Output path for preference dataset
        primary_metric: Metric to use for preference selection
    """

    print(f"Loading evaluation results from {scores_file}")
    scores = load_evaluation_results(scores_file)

    print(f"Loading prompts from {prompts_file}")
    prompts = load_prompts(prompts_file)

    print(f"Grouping images by prompt index...")
    grouped_images = group_images_by_prompt_index(scores)

    preference_pairs = []
    skipped_groups = 0

    print(f"Selecting preference pairs using metric: {primary_metric}")

    for prompt_index, images in grouped_images.items():
        if prompt_index >= len(prompts):
            print(
                f"Warning: Prompt index {prompt_index} exceeds prompts length {len(prompts)}")
            continue

        pair = select_preference_pair(images, primary_metric)
        if pair is None:
            skipped_groups += 1
            continue

        (best_filename, best_scores), (worst_filename, worst_scores) = pair

        # Construct full image paths
        best_image_path = os.path.join(images_dir, best_filename)
        worst_image_path = os.path.join(images_dir, worst_filename)

        # Check if image files exist
        if not os.path.exists(best_image_path):
            print(f"Warning: Image not found: {best_image_path}")
            continue
        if not os.path.exists(worst_image_path):
            print(f"Warning: Image not found: {worst_image_path}")
            continue

        # Convert images to bytes
        try:
            best_image_bytes = image_to_bytes(best_image_path)
            worst_image_bytes = image_to_bytes(worst_image_path)
        except Exception as e:
            print(f"Error reading images for prompt {prompt_index}: {e}")
            continue

        # Calculate composite scores for metadata
        best_composite_score = calculate_composite_score(
            best_scores, primary_metric)
        worst_composite_score = calculate_composite_score(
            worst_scores, primary_metric)

        # Create preference pair
        # jpg_0 is the better image (preferred), jpg_1 is worse
        # label_0 = 1 means jpg_0 is preferred
        preference_pair = {
            "jpg_0": best_image_bytes,
            "jpg_1": worst_image_bytes,
            "label_0": 1,  # jpg_0 (best) is preferred
            "caption": prompts[prompt_index],
            "metadata": {
                "prompt_index": prompt_index,
                "best_filename": best_filename,
                "worst_filename": worst_filename,
                "best_scores": best_scores,
                "worst_scores": worst_scores,
                "primary_metric": primary_metric,
                "best_composite_score": best_composite_score,
                "worst_composite_score": worst_composite_score,
                "score_difference": best_composite_score - worst_composite_score
            }
        }

        preference_pairs.append(preference_pair)

        if len(preference_pairs) % 100 == 0:
            print(f"Processed {len(preference_pairs)} preference pairs...")

    print(f"\nDataset Statistics:")
    print(f"  Total prompt groups: {len(grouped_images)}")
    print(f"  Skipped groups (insufficient variation): {skipped_groups}")
    print(f"  Generated preference pairs: {len(preference_pairs)}")

    if len(preference_pairs) == 0:
        print("No preference pairs generated. Check your data and criteria.")
        return

    # Save dataset
    print(f"\nSaving preference dataset to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'wb') as f:
        # Save as binary format to handle image bytes
        import pickle
        pickle.dump(preference_pairs, f)

    # Also save a readable metadata file
    metadata_file = output_file.replace('.pkl', '_metadata.json')
    metadata = []
    for pair in preference_pairs:
        meta = pair["metadata"].copy()
        # Remove image bytes for readability
        metadata.append(meta)

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Dataset saved to: {output_file}")
    print(f"Metadata saved to: {metadata_file}")

    # Print some statistics about score differences
    score_diffs = [pair["metadata"]["score_difference"]
                   for pair in preference_pairs]
    print(f"\nScore Difference Statistics:")
    print(f"  Mean: {sum(score_diffs) / len(score_diffs):.4f}")
    print(f"  Min: {min(score_diffs):.4f}")
    print(f"  Max: {max(score_diffs):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Build preference dataset from evaluation results")
    parser.add_argument("--scores_file", type=str, required=True,
                        help="Path to evaluation scores JSON file")
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="Path to prompts text file")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing generated images")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output path for preference dataset (will be .pkl format)")
    parser.add_argument("--primary_metric", type=str, default="semantic_coverage",
                        choices=["semantic_coverage", "relation_validity", "style_score",
                                 "semantic_coverage+relation_validity",
                                 "semantic_coverage+style_score",
                                 "relation_validity+style_score",
                                 "semantic_coverage+relation_validity+style_score"],
                        help="Primary metric for preference selection. Supports single metrics or composite metrics (e.g., 'semantic_coverage+relation_validity' for average of both)")

    args = parser.parse_args()

    # Ensure output file has .pkl extension
    if not args.output_file.endswith('.pkl'):
        args.output_file += '.pkl'

    build_preference_dataset(
        scores_file=args.scores_file,
        prompts_file=args.prompts_file,
        images_dir=args.images_dir,
        output_file=args.output_file,
        primary_metric=args.primary_metric
    )


if __name__ == "__main__":
    main()
