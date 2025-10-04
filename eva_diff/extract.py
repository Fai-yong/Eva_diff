import argparse
import json
from tqdm import tqdm
import extract_sym_func as esf


def get_prompts(file_path):
    """Load prompts from text file"""
    with open(file_path, "r", encoding="utf-8") as f:
        prompts = f.readlines()
        prompts = [prompt.strip() for prompt in prompts]
    return prompts


def extract_templates(prompts_file, output_file):
    """Extract semantic templates from prompts"""
    prompts = get_prompts(prompts_file)
    template_list = []

    print(f"Processing {len(prompts)} prompts...")

    # Extract objects, attributes, and relations from prompts
    for index, prompt in enumerate(tqdm(prompts)):
        # Parse prompt for semantic elements
        parsed = esf.parse_prompt(prompt)

        # Generate evaluation template
        template = esf.generate_eval_template(parsed, prompt)

        template_item = {
            "index": index,
            "prompt": prompt,
            "template": template
        }
        template_list.append(template_item)

    # Save templates to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(template_list, f, indent=2, ensure_ascii=False)

    print(f"Templates saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract semantic templates from text prompts")
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="Path to input prompts text file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to output JSON file for templates")

    args = parser.parse_args()

    extract_templates(args.prompts_file, args.output_file)


if __name__ == "__main__":
    main()
