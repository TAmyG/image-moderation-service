import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

SUSPICIOUS_TEXTS = [
    "a violent scene",
    "a person holding a gun",
    "blood or injury",
    "a sexual situation",
    "nudity",
    "drug use"
]

def is_semantically_inappropriate(image_path: str) -> bool:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=SUSPICIOUS_TEXTS, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    max_prob = torch.max(probs).item()
    return max_prob > 0.75
