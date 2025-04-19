import torch
import clip
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
import shutil
import os

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#NSFW_LABELS = ["nsfw", "porn", "nudity", "sexy", "explicit"]
#SAFE_LABELS = ["clothed", "wholesome", "nature", "cat", "dog"]

VIOLENCE_LABELS = ["gun", "knife", "blood", "fight", "violence", "explosion", "dead body", "injury", "weapon", "war"]
SAFE_LABELS = ["no weapon", "peace", "group of friends", "landscape", "sports", "smiling people"]


@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        image_path = tmp.name

    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # Combine labels and tokenize
        #labels = NSFW_LABELS + SAFE_LABELS
        labels = VIOLENCE_LABELS + SAFE_LABELS
        text = clip.tokenize(labels).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).squeeze(0)

        scores = list(zip(labels, similarity.tolist()))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        return {
            "top_label": sorted_scores[0][0],
            "scores": [{ "label": l, "score": round(s, 4)} for l, s in sorted_scores]
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
