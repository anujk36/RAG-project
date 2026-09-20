from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.routers import health, ask, documents, images, upload

from app.logging_config import setup_logging, logger

setup_logging()

app = FastAPI(
    title="RAG API",
    description="A Retrieval-Augmented Generation API built with LangChain, LangGraph, and Google Gemini.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ask.router)
app.include_router(documents.router)
app.include_router(images.router)
app.include_router(upload.router)


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse("index.html")


logger.info("RAG API startup complete.")
