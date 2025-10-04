import argparse
import json
import os
from tqdm import tqdm
import eval_func


def get_results(file_base, results_tag):
    """Load evaluation results from JSON file"""
    results_file = f"{file_base}/{results_tag}_eval_results.json"
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results


def save_scores(file_base, all_metrics, results_tag):
    """Save calculated metrics to JSON file"""
    os.makedirs(file_base, exist_ok=True)
    scores_file = f"{file_base}/{results_tag}_scores.json"
    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"Scores saved to {scores_file}")


def calculate_metrics(results_base, results_tag):
    """Calculate evaluation metrics for a specific model's results"""
    # Load evaluation results
    results = get_results(file_base=results_base, results_tag=results_tag)
    all_metrics = {}

    print(f"Calculating metrics for {results_tag} in {results_base}...")

    # Calculate metrics for each response
    for key, value in tqdm(results.items()):
        if value is None:
            continue

        try:
            metrics = eval_func.calculate_hybrid_metrics(value)
            all_metrics[key] = metrics
        except Exception as e:
            print(f"Error evaluating {key}: {results_tag}, {results_base}")
            print(f"Error details: {e}")
            continue

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Calculate evaluation metrics from MLLM results")
    parser.add_argument("--results_base", type=str, required=True,
                        help="Base directory containing evaluation results")
    parser.add_argument("--results_tags", type=str, nargs="+", required=True,
                        help="List of model tags to evaluate (e.g., llava-1_6 llava-ov qwen2_5-vl)")

    args = parser.parse_args()

    # Process each model tag
    for tag in args.results_tags:
        print(f"\nProcessing {tag}...")

        try:
            all_metrics = calculate_metrics(
                results_base=args.results_base,
                results_tag=tag
            )

            # Save metrics
            save_scores(
                file_base=args.results_base,
                all_metrics=all_metrics,
                results_tag=tag
            )

            print(f"Processed {len(all_metrics)} items for {tag}")

        except Exception as e:
            print(f"Error processing {tag}: {e}")
            continue


if __name__ == "__main__":
    main()
