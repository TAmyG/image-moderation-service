from fastapi import FastAPI, UploadFile, File
from utils import save_upload_file
from nsfw_classifier import is_nsfw
from clip_semantic import is_semantically_inappropriate

app = FastAPI()

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    path = await save_upload_file(file)

    # Paso 1: Clasificación NSFW
    if is_nsfw(path):
        return {"status": "rejected", "reason": "NSFW detected"}

    # Paso 2: Detección semántica (CLIP)
    if is_semantically_inappropriate(path):
        return {"status": "rejected", "reason": "Semantic content violation"}

    return {"status": "approved"}
