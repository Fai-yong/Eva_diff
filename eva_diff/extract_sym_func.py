import spacy
from spacy.matcher import Matcher
import json

# Initialize NLP models
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Predefined art movement lexicon
PREDEFINED_ART_MOVEMENTS = {"renaissance",
                            "cyberpunk", "vaporwave", "brutalist"}


def parse_prompt(prompt_text):
    """
    Multi-dimensional semantic parser for image generation prompts
    Extracts objects, attributes, relations, and styles from text prompts
    """
    doc = nlp(prompt_text)
    elements = {
        "objects": [],
        "attributes": {},
        "relations": [],
        "styles": {
            "artists": [],
            "movements": [],
            "mediums": [],
            "aesthetics": []
        }
    }

    # Rule 1: Extract compound objects (e.g., "anthropomorphic anthro male fox")
    compound_pattern = [
        {"POS": "ADJ"},
        {"POS": "ADJ", "OP": "*"},
        {"POS": "NOUN"},
        {"POS": "NOUN", "OP": "+"}
    ]
    matcher.add("COMPOUND_OBJECT", [compound_pattern])
    matches = matcher(doc)

    for match_id, start, end in matches:
        span = doc[start:end]
        elements["objects"].append({"text": span.text, "type": "compound"})

    # Rule 2: Extract artist references (e.g., "by Gustav Klimt")
    artist_pattern = [{"LOWER": "by"}, {"POS": "PROPN", "OP": "+"}]
    matcher.add("ARTIST_REF", [artist_pattern])
    matches = matcher(doc)

    for match_id, start, end in matches:
        elements["styles"]["artists"].append(doc[start+1:end].text)

    # Extract meaningful verb relations using spaCy
    # Filter out meaningless verbs
    meaningless_verbs = {"be", "is", "are", "was", "were", "been", "being",
                         "have", "has", "had", "having",
                         "do", "does", "did", "doing", "done",
                         "will", "would", "shall", "should", "could", "can", "may", "might", "must"}

    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() not in meaningless_verbs:
            elements["relations"].append({
                "verb": token.text,
                "args": f"Action: {token.text}"
            })

    return elements


def generate_eval_template(parsed_data, original_prompt):
    """
    Generate evaluation template based on parsed prompt data
    Creates structured questions for image-text alignment evaluation
    """
    questions = []

    # Generate object-related questions
    for obj in parsed_data["objects"]:
        obj_text = obj["text"]
        questions.append(
            f"Is there a '{obj_text}' in the image? Describe its key attributes.")

    # Generate style-related questions
    if parsed_data["styles"]["artists"]:
        artists = ", ".join(parsed_data["styles"]["artists"])
        questions.append(
            f"Does the image reflect the style of {artists}? Provide specific evidence.")

    # Generate relation-related questions
    for rel in parsed_data["relations"]:
        questions.append(
            f"Does the action '{rel['verb']}' appear in the image? Describe how it's represented.")

    # Create evaluation template
    template = {
        "instructions": (
            f"You are a vision-language reasoning expert. This image was generated from the following prompt: '{original_prompt}'. "
            f"Please analyze the image and evaluate how well it matches the original prompt by answering the following questions. "
            f"After answering internally, output a structured JSON result following the format shown in 'response_example'. "
            f"Only output the final JSON without extra explanations."
        ),
        "questions": questions,
        "response_example": {
            "objects": [
                {
                    "name": "sample_object_1",
                    "present": True,
                    "attributes": {
                        "attribute_1": True,
                        "attribute_2": False
                    }
                }
            ],
            "style_consistency": {
                "score": 3,
                "evidence": "Visual elements follow the referenced art style with moderate consistency."
            },
            "relations": [
                {
                    "verb": "sample_action_1",
                    "valid": True
                }
            ]
        }
    }

    return template
