import argparse
import os
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
from tqdm import tqdm
from typing import List, Optional, Union


class ImageGenerator:
    """
    Unified image generator class for different model types (untuned, LoRA, tuned)
    """

    def __init__(self, model_path: str, model_type: str = 'untuned',
                 lora_path: Optional[str] = None, base_model_path: Optional[str] = None,
                 device: str = "cuda:0"):
        """
        Initialize the image generator with specified model configuration

        Args:
            model_path: Path to the main model
            model_type: Type of model ('untuned', 'lora', 'tuned')
            lora_path: Path to LoRA weights (required for 'lora' type)
            base_model_path: Path to base model (required for 'tuned' type)
            device: Device to use for generation
        """
        self.model_path = model_path
        self.model_type = model_type
        self.lora_path = lora_path
        self.base_model_path = base_model_path
        self.device = device
        self.pipeline = None
        self.prompts = None

        # Validate arguments
        if model_type == 'lora' and not lora_path:
            raise ValueError(
                "lora_path is required when using lora model type")
        if model_type == 'tuned' and not base_model_path:
            raise ValueError(
                "base_model_path is required when using tuned model type")

        # Initialize pipeline
        self._load_pipeline()

    def _load_pipeline(self):
        """Load the appropriate pipeline based on model type"""
        if self.model_type == 'tuned':
            # Load fine-tuned components
            unet = UNet2DConditionModel.from_pretrained(
                f'{self.model_path}/unet')
            text_encoder = CLIPTextModel.from_pretrained(
                f'{self.model_path}/text_encoder')

            # Load pipeline with fine-tuned components
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.base_model_path,
                unet=unet,
                text_encoder=text_encoder,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            # Load standard pipeline (untuned or lora)
            try:
                # Try loading with fp16 variant first (for pretrained models)
                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    variant="fp16",
                    use_safetensors=True
                ).to(self.device)
            except ValueError as e:
                if "variant=fp16" in str(e):
                    # Fallback: load without variant (for fine-tuned models)
                    print("fp16 variant not found, loading standard model...")
                    self.pipeline = DiffusionPipeline.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        safety_checker=None,
                        use_safetensors=True
                    ).to(self.device)
                else:
                    raise e

            # Load LoRA weights if specified
            if self.model_type == 'lora' and self.lora_path:
                self.pipeline.load_lora_weights(self.lora_path)

        # Enable memory efficient attention
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
            print("xformers memory efficient attention enabled")
        except Exception as e:
            print(f"Could not enable xformers: {e}")

    def load_prompts(self, prompts_path: str):
        """Load prompts from text file"""
        with open(prompts_path, "r", encoding="utf-8") as f:
            self.prompts = [prompt.strip() for prompt in f.readlines()]
        print(f"Loaded {len(self.prompts)} prompts")
        return self.prompts

    def set_prompts(self, prompts: List[str]):
        """Set prompts directly from a list"""
        self.prompts = prompts
        print(f"Set {len(self.prompts)} prompts")

    def generate_images(self, save_path: str, inference_steps: int = 50,
                        guidance_scale: float = 7.5, num_images_per_prompt: int = 1,
                        seed: Optional[int] = None, test_mode: bool = False):
        """
        Generate images using the loaded pipeline and prompts

        Args:
            save_path: Directory to save generated images
            inference_steps: Number of inference steps
            guidance_scale: Guidance scale for generation
            num_images_per_prompt: Number of images to generate per prompt
            seed: Fixed seed for reproducible generation (if None, no seed is set)
            test_mode: If True, generate only one test image
        """
        if self.prompts is None:
            raise ValueError(
                "Prompts not loaded. Call load_prompts() or set_prompts() first.")

        # Create save directory
        os.makedirs(save_path, exist_ok=True)

        if test_mode:
            self._generate_test_images(save_path, inference_steps, guidance_scale,
                                       num_images_per_prompt, seed)
        else:
            self._generate_all_images(save_path, inference_steps, guidance_scale,
                                      num_images_per_prompt, seed)

    def _generate_test_images(self, save_path: str, inference_steps: int,
                              guidance_scale: float, num_images_per_prompt: int,
                              seed: Optional[int]):
        """Generate single test image"""
        test_dir = os.path.join(save_path, 'test')
        os.makedirs(test_dir, exist_ok=True)

        test_index = 5
        if test_index >= len(self.prompts):
            test_index = 0

        # Set up generator if seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Using fixed seed {seed} for test image")

        images = self.pipeline(
            self.prompts[test_index],
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator
        ).images

        for j, image in enumerate(images):
            test_filename = f'test_{test_index}_{j}.png'
            image.save(os.path.join(test_dir, test_filename))
            print(f"Test image saved: {os.path.join(test_dir, test_filename)}")

    def _generate_all_images(self, save_path: str, inference_steps: int,
                             guidance_scale: float, num_images_per_prompt: int,
                             seed: Optional[int]):
        """Generate all images"""
        # Set up generator once if seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Using fixed seed {seed} for all images")

        for i, prompt in tqdm(enumerate(self.prompts), desc="Generating images", total=len(self.prompts)):
            images = self.pipeline(
                prompt,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator
            ).images

            for j, image in enumerate(images):
                image.save(os.path.join(save_path, f"{i}_{j}.png"))


def get_prompts(prompts_path):
    """Load prompts from text file (legacy function for backward compatibility)"""
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = f.readlines()
        prompts = [prompt.strip() for prompt in prompts]
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion")
    parser.add_argument('--prompts_file', type=str, required=True,
                        help='Path to prompts text file')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory to save generated images')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to diffusion model')
    parser.add_argument('--model_type', type=str, choices=['untuned', 'lora', 'tuned'],
                        default='untuned', help='Type of model to use')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Path to LoRA weights (required for lora model type)')
    parser.add_argument('--base_model_path', type=str, default=None,
                        help='Path to base model (required for tuned model type)')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='Device to use for generation')
    parser.add_argument('--inference_steps', type=int, default=50,
                        help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Guidance scale for generation')
    parser.add_argument('--test', action='store_true',
                        help='Generate single test image instead of all images')
    parser.add_argument('--num_images_per_prompt', type=int, default=1,
                        help='Number of images to generate per prompt')
    parser.add_argument('--seed', type=int, default=None,
                        help='Fixed seed for reproducible generation (all prompts use the same seed)')

    args = parser.parse_args()

    # Initialize ImageGenerator
    print(f"Initializing {args.model_type} model from {args.model_path}")
    generator = ImageGenerator(
        model_path=args.model_path,
        model_type=args.model_type,
        lora_path=args.lora_path,
        base_model_path=args.base_model_path,
        device=args.device
    )

    # Load prompts
    generator.load_prompts(args.prompts_file)

    # Generate images
    generator.generate_images(
        save_path=args.save_path,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        seed=args.seed,
        test_mode=args.test
    )

    if not args.test:
        total_images = len(generator.prompts) * args.num_images_per_prompt
        print(
            f"Generated {total_images} images ({args.num_images_per_prompt} per prompt) and saved to {args.save_path}")
        if args.seed is not None:
            print('='*100)
            print(
                f"Used fixed seed: {args.seed} (all prompts used the same seed)")
            print('='*100)
        else:
            print("No seed specified - generation was random")
    else:
        print("Test generation completed")


if __name__ == "__main__":
    main()
