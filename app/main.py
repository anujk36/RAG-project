from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.logging_config import setup_logging, logger
from app.routers import health, ask

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

logger.info("RAG API startup complete.")
