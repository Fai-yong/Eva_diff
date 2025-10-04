import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import eval_func
import mllm_func


def load_result_json(json_path):
    """Load existing results from JSON file"""
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_result_json(json_path, result_data):
    """Save results to JSON file"""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)


def get_error_images(result_data):
    """Get list of images that failed evaluation"""
    return [img_name for img_name, val in result_data.items() if val is None]


def generate_response_for_image(image_path, template, model, processor, device):
    """Generate model response for a single image"""
    try:
        image = Image.open(image_path)
        conversation = eval_func.build_mllm_conversation(template)
        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True)

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(device, torch.float16)

        output = model.generate(**inputs, max_new_tokens=4000)
        response = processor.decode(output[0], skip_special_tokens=True)

        json_response = eval_func.get_json_resp(response)
        return json_response
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def evaluate_images(images_dir, template_list, model, processor, device, result_file, start_index=0, end_index=None):
    """Evaluate a batch of images"""
    result_data = load_result_json(result_file)

    # Get list of image files
    image_files = [f for f in os.listdir(
        images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Custom sorting function to handle both "0.png" and "0_1.png" formats
    def extract_prompt_index(filename):
        """Extract prompt index from filename (e.g., '0_1.png' -> 0, '5.png' -> 5)"""
        name_without_ext = filename.split('.')[0]
        if '_' in name_without_ext:
            # Extract prompt index from "0_1"
            return int(name_without_ext.split('_')[0])
        else:
            return int(name_without_ext)  # Handle legacy "0.png" format

    # Sort by prompt index, then by image variant
    image_files.sort(key=lambda x: (extract_prompt_index(x), x))

    if end_index is None:
        end_index = len(image_files)

    # Process images in specified range
    processed_count = 0
    for i in tqdm(range(len(image_files)), desc="Evaluating images"):
        if processed_count < start_index:
            processed_count += 1
            continue
        if processed_count >= end_index:
            break

        image_file = image_files[i]
        image_path = os.path.join(images_dir, image_file)

        # Skip if already processed
        if image_file in result_data and result_data[image_file] is not None:
            processed_count += 1
            continue

        # Extract prompt index from filename
        prompt_index = extract_prompt_index(image_file)

        # Check if prompt index is within template range
        if prompt_index >= len(template_list):
            print(
                f"Warning: Prompt index {prompt_index} for {image_file} exceeds template list length {len(template_list)}")
            processed_count += 1
            continue

        # Use prompt index to get corresponding template
        template = template_list[prompt_index]
        json_response = generate_response_for_image(
            image_path, template, model, processor, device
        )

        result_data[image_file] = json_response

        # Save periodically
        if processed_count % 10 == 0:
            save_result_json(result_file, result_data)

        processed_count += 1

    # Final save
    save_result_json(result_file, result_data)
    return result_data


def regenerate_missing_results(result_file, images_dir, template_list, model, processor, device):
    """Regenerate results for failed images"""
    result_data = load_result_json(result_file)
    bad_image_keys = get_error_images(result_data)

    if not bad_image_keys:
        print("No missing results to regenerate.")
        return {"total": 0}

    error_dict = {"total": 0}

    # Helper function to extract prompt index (same as in evaluate_images)
    def extract_prompt_index(filename):
        """Extract prompt index from filename (e.g., '0_1.png' -> 0, '5.png' -> 5)"""
        name_without_ext = filename.split('.')[0]
        if '_' in name_without_ext:
            # Extract prompt index from "0_1"
            return int(name_without_ext.split('_')[0])
        else:
            return int(name_without_ext)  # Handle legacy "0.png" format

    for image_key in tqdm(bad_image_keys, desc="Regenerating missing results"):
        error_dict["total"] += 1

        image_path = os.path.join(images_dir, image_key)

        # Extract prompt index from filename
        prompt_index = extract_prompt_index(image_key)

        if prompt_index >= len(template_list):
            print(
                f"Prompt index {prompt_index} for {image_key} exceeds template list length {len(template_list)}")
            continue

        template = template_list[prompt_index]

        # Try multiple times for failed images
        max_retries = 3
        for retry in range(max_retries):
            json_response = generate_response_for_image(
                image_path, template, model, processor, device
            )

            if json_response is not None:
                result_data[image_key] = json_response
                error_dict[image_key] = retry + 1
                break
        else:
            error_dict[image_key] = max_retries

    save_result_json(result_file, result_data)
    return error_dict


def main():
    parser = argparse.ArgumentParser(
        description="Run MLLM evaluation on images")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Type of model to use (llava, qwen2_vl, qwen2_5_vl, gemma3, etc.)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model directory")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing images to evaluate")
    parser.add_argument("--template_file", type=str, required=True,
                        help="Path to evaluation template JSON file")
    parser.add_argument("--output_file", type=str, required=False,
                        help="Path to output results JSON file (deprecated, use --save_dir and --file_name)")
    parser.add_argument("--save_dir", type=str, required=False,
                        help="Directory to save output results")
    parser.add_argument("--file_name", type=str, required=False,
                        help="Output file name (with .json extension)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for inference")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Starting index for batch processing")
    parser.add_argument("--end_index", type=int, default=None,
                        help="Ending index for batch processing")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate missing results only")

    args = parser.parse_args()

    # Handle output file path
    if args.save_dir and args.file_name:
        # Create save directory if it doesn't exist
        os.makedirs(args.save_dir, exist_ok=True)
        output_file = os.path.join(args.save_dir, args.file_name)
    elif args.output_file:
        # Use legacy output_file parameter
        output_file = args.output_file
        # Create directory for output_file if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    else:
        raise ValueError(
            "Either provide --output_file OR both --save_dir and --file_name")

    # Load model and processor
    print(f"Loading {args.model_type} model from {args.model_path}")
    device = torch.device(args.device)

    if args.model_type == "llava":
        model = mllm_func.load_llava_model(args.model_path)
        model.to(device)
    elif args.model_type == "gemma3":
        model = mllm_func.load_gemma3_model(args.model_path)
        model.to(device)
    elif args.model_type == "qwen2-5-vl":
        model = mllm_func.load_qwen2_5_vl_model(args.model_path)
        model.to(device)
    else:
        model = mllm_func.load_model(args.model_type, args.model_path)

    processor = mllm_func.load_processor(args.model_path)

    # Load evaluation templates
    template_list = eval_func.get_template_list(args.template_file)

    if args.regenerate:
        # Regenerate missing results
        error_dict = regenerate_missing_results(
            result_file=output_file,
            images_dir=args.images_dir,
            template_list=template_list,
            model=model,
            processor=processor,
            device=device
        )

        # Save error summary
        error_summary_file = output_file.replace(
            ".json", "_error_summary.json")
        with open(error_summary_file, "w", encoding="utf-8") as f:
            json.dump(error_dict, f, indent=2)

        print(f"Regeneration complete. Error count: {error_dict['total']}")
    else:
        # Regular evaluation
        result_data = evaluate_images(
            images_dir=args.images_dir,
            template_list=template_list,
            model=model,
            processor=processor,
            device=device,
            result_file=output_file,
            start_index=args.start_index,
            end_index=args.end_index
        )

        print(f"Evaluation complete. Results saved to {output_file}")
        print(f"Processed {len(result_data)} images")


if __name__ == "__main__":
    main()
