import shutil
from pathlib import Path
from fastapi import UploadFile

async def save_upload_file(upload_file: UploadFile) -> str:
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / upload_file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return str(file_path)
