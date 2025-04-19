from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

# Modelo OpenNSFW2 de LAION
model = AutoModelForImageClassification.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
extractor = AutoFeatureExtractor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")

def is_nsfw(image_path: str) -> bool:
    image = Image.open(image_path).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits[0], dim=0)
    porn_score = scores[433]  # Ejemplo: ID correspondiente a categoría sensible (puedes ajustarlo)

    return porn_score.item() > 0.8
