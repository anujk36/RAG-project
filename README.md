# RAG Project

A full-stack Retrieval-Augmented Generation (RAG) system built from scratch to answer questions from custom documents (PDF, TXT, DOCX), using Google Gemini, LangChain, LangGraph, LangSmith, and FastAPI.

## Features
- Multi-format document ingestion (PDF, TXT, DOCX)
- Semantic search via vector embeddings (ChromaDB)
- Conditional retrieval logic — rejects off-topic questions instead of hallucinating
- Source citation (file + page number) for every answer
- Full observability via LangSmith tracing
- Structured FastAPI backend: config management, logging, custom exceptions, input validation
- REST API with interactive docs (`/docs`)
- Simple web frontend with live backend health check

## Tech Stack
- **AI/ML**: LangChain, LangGraph, Google Gemini (LLM + embeddings), ChromaDB
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: HTML, JavaScript, Tailwind CSS
- **Observability**: LangSmith
- **Version Control**: Git/GitHub

## Project Structure
rag-project/
├── app/
│ ├── main.py # FastAPI app entrypoint
│ ├── config.py # centralized settings (reads .env)
│ ├── logging_config.py # logging setup
│ ├── exceptions.py # custom exception types
│ ├── rag_service.py # RAG/LangGraph logic
│ ├── models/schemas.py # request/response data shapes
│ └── routers/ # /ask, /health, /documents endpoints
├── build_index.py # builds the vector store from documents/
├── documents/ # source documents (PDF, TXT, DOCX)
├── index.html # web frontend
└── requirements.txt


## Setup

1. Clone the repo and create a virtual environment:
python -m venv venv
venv\Scripts\activate


2. Install dependencies:
pip install -r requirements.txt


3. Copy `.env.example` to `.env` and fill in your API keys.

4. Add your documents to the `documents/` folder.

5. Build the index:
python build_index.py


6. Run the backend:
uvicorn app.main:app --reload


7. Open `index.html` in your browser (or serve it via Live Server) to use the web interface.

## API Endpoints

- `GET /health` — check if the service is running
- `POST /ask` — ask a question, returns a grounded answer with context
- `GET /documents` — list currently indexed documents

Interactive API documentation available at `http://127.0.0.1:8000/docs`.

## How It Works
1. Documents are loaded and split into chunks.
2. Chunks are embedded and stored in a vector database.
3. A user's question is embedded and matched against stored chunks.
4. If no relevant match is found, the system declines to answer rather than guessing.
5. Relevant chunks are passed to Gemini to generate a grounded, cited answer.