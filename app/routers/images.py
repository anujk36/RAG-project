from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings

router = APIRouter()

DOCUMENTS_DIR = Path(settings.documents_dir).resolve()
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@router.get("/images/{image_path:path}", tags=["Images"])
def get_image(image_path: str):
    requested = (DOCUMENTS_DIR / image_path).resolve()

    if not requested.is_relative_to(DOCUMENTS_DIR):
        raise HTTPException(status_code=400, detail="Invalid path")

    if requested.suffix.lower() not in ALLOWED_EXTENSIONS or not requested.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(requested)
