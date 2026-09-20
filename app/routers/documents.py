from fastapi import APIRouter
from pydantic import BaseModel
from app.config import settings
import os

router = APIRouter()


class DocumentListResponse(BaseModel):
    documents: list[str]
    count: int


@router.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents():
    documents_dir = settings.documents_dir
    if not os.path.exists(documents_dir):
        return DocumentListResponse(documents=[], count=0)

    files = [
        f for f in os.listdir(documents_dir)
        if os.path.isfile(os.path.join(documents_dir, f)) and not f.startswith(".")
    ]
    return DocumentListResponse(documents=files, count=len(files))