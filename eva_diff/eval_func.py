import torch
from PIL import Image
import re
import os
import json


def get_prompts(file_path):
    """Load prompts from text file"""
    with open(file_path, "r", encoding="utf-8") as f:
        prompts = f.readlines()
        prompts = [prompt.strip() for prompt in prompts]
    return prompts


def get_template_list(template_file):
    """Load evaluation templates from JSON file"""
    template_list = []
    with open(template_file, "r", encoding="utf-8") as f:
        mllm_prompts = json.load(f)
        for prompt in mllm_prompts:
            # Handle both 'template' and 'llava_template' field names for backward compatibility
            if "template" in prompt:
                template_list.append(prompt["template"])
            elif "llava_template" in prompt:
                template_list.append(prompt["llava_template"])
            else:
                raise KeyError(
                    f"Template field not found. Expected 'template' or 'llava_template' in: {prompt.keys()}")
    return template_list


def clean_response(text, prompt):
    """Clean model response by removing unnecessary information"""
    # Remove user/assistant tags
    text = re.sub(r"user\s*\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"assistant\s*\n", "", text, flags=re.IGNORECASE)

    # Remove the prompt from the response
    text = text.replace(prompt, "").strip()
    return text


def build_mllm_conversation(template):
    """Build conversation format for multimodal LLM"""
    conversation = []

    # Add initial instruction with image
    conversation.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": template["instructions"]}
        ]
    })

    # Add evaluation questions
    for q in template["questions"]:
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": q}
            ]
        })

    # Add final JSON format instruction
    conversation.append({
        "role": "user",
        "content": [
            {"type": "text", "text": (
                "Please analyze this image and output the results in the following JSON format:\n"
                "Do not include any additional explanations. Make sure the JSON output is valid:\n"
                f"{json.dumps(template['response_example'], indent=4)}"
            )}
        ]
    })

    return conversation


def ask_model(image, prompt, model, processor, device):
    """Ask question to the model and return response"""
    conversation = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]

    prompt_template = processor.apply_chat_template(
        conversation, add_generation_prompt=True)

    inputs = processor(
        images=image,
        text=prompt_template,
        return_tensors="pt"
    ).to(device, torch.float16)

    # Generate response
    output = model.module.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )

    raw_answer = processor.decode(output[0], skip_special_tokens=True)
    return raw_answer


def calculate_hybrid_metrics(llm_json_output, image_path=None, original_prompt=None):
    """
    Calculate comprehensive evaluation metrics including:
    - Semantic coverage: How well objects are detected
    - Relation validity: Accuracy of relationship detection
    - Style consistency: Alignment with artistic style
    """
    eval_data = llm_json_output

    # Semantic coverage calculation
    total_elements = 0
    matched_elements = 0

    # Count detected objects
    o_num = len(eval_data["objects"]) if "objects" in eval_data else 0

    for obj in eval_data["objects"]:
        if obj["present"]:
            matched_elements += 1
            # Check attribute matches
            if "attributes" in obj:
                total_elements += len(obj["attributes"])
                matched_attrs = sum(
                    1 for attr_value in obj["attributes"].values() if attr_value)
                matched_elements += matched_attrs
        total_elements += 1

    semantic_coverage = matched_elements / \
        total_elements if total_elements > 0 else 0

    # Relation validity calculation
    if "relations" in eval_data:
        valid_relations = sum(
            1 for rel in eval_data["relations"] if rel["valid"])
        r_num = len(eval_data["relations"])
        relation_validity = valid_relations / r_num if r_num > 0 else 0
    else:
        relation_validity = 0

    # Style consistency (normalized to 0-1)
    style_score = eval_data["style_consistency"]["score"] / 5

    return {
        "semantic_coverage": round(semantic_coverage, 4),
        "relation_validity": round(relation_validity, 4),
        "style_score": round(style_score, 4),
        "object_num": o_num,
        "total_attrs": total_elements
    }


def get_json_resp(response):
    """Extract JSON response from model output"""
    match = re.search(r'```json\s*(\{.*\})\s*```', response, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            parsed_output = json.loads(json_str)
            return parsed_output
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
    else:
        print("No JSON block found in response")
        return None


# Cultural reference detection
CULTURAL_LEXICON = {
    "isaac asimov's foundation": {"type": "literary_work", "tags": ["sci-fi", "library"]},
    "borderlands": {"type": "video_game", "tags": ["cell-shaded", "post-apocalyptic"]}
}


# def detect_cultural_references(prompt_text):
#     """Detect cultural/fictional entity references in prompt"""
#     detected = []
#     nlp = spacy.load("en_core_web_lg")
#     doc = nlp(prompt_text)

#     # Sliding window detection for multi-word entities
#     for i in range(len(doc)):
#         for j in range(i+1, min(i+5, len(doc))):
#             phrase = doc[i:j].text.lower()
#             if phrase in CULTURAL_LEXICON:
#                 detected.append({
#                     "phrase": phrase,
#                     "metadata": CULTURAL_LEXICON[phrase]
#                 })

#     return detected


def extract_evidence(text, keywords):
    """Extract evidence sentences based on keywords"""
    sentences = re.split(r'[.!?]', text)
    for sent in sentences:
        if any(kw.lower() in sent.lower() for kw in keywords):
            return sent.strip()
    return "No explicit mention found"
