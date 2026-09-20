from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    google_api_key: str
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-project"

    llm_model: str = "gemini-3.6-flash"
    embedding_model: str = "gemini-embedding-2-preview"
    chroma_persist_dir: str = "./chroma_db"
    relevance_threshold: float = 0.7
    retrieval_k: int = 3

    class Config:
        env_file = ".env"

settings = Settings()