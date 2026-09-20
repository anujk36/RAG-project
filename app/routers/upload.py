from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.exceptions import RAGException
from app.logging_config import logger
from app.rag_service import rag_service

router = APIRouter()

DOCUMENTS_DIR = Path(settings.documents_dir)
PDF_MAGIC = b"%PDF-"


class UploadResponse(BaseModel):
    filename: str
    text_chunks: int
    images: int


@router.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest_path = DOCUMENTS_DIR / filename
    if dest_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"'{filename}' already exists. Rename the file or remove the existing one first.",
        )

    contents = await file.read(settings.max_upload_bytes + 1)
    if len(contents) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File exceeds the {limit_mb}MB upload limit.")

    if not contents.startswith(PDF_MAGIC):
        raise HTTPException(status_code=400, detail="File does not look like a valid PDF.")

    DOCUMENTS_DIR.mkdir(exist_ok=True)
    dest_path.write_bytes(contents)

    try:
        result = rag_service.add_pdf(dest_path)
    except RAGException as e:
        dest_path.unlink(missing_ok=True)
        logger.error(f"Indexing failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail="Uploaded, but failed to index the document.")

    return UploadResponse(filename=filename, text_chunks=result["text_chunks"], images=result["images"])
