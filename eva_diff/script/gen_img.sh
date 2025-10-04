export NVTE_FRAMEWORK=pytorch
export NVTE_DISABLE_TE=1

MODEL_PATH="path/to/pretrained-model/sd_xl"
PROMPTS_FILE="path/to/prompts.txt"
OUTPUT_DIR="path/to/image_output/sd_xl/untuned"

python path/to/sd.py \
  --prompts_file $PROMPTS_FILE \
  --save_path $OUTPUT_DIR \
  --model_type untuned \
  --model_path $MODEL_PATH \
  --device cuda:0 \
  --inference_steps 50 \
  --guidance_scale 7.5 \
  --num_images_per_prompt 4 \
  --seed 42
