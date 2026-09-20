# RAG Project

A Retrieval-Augmented Generation (RAG) system built from scratch to answer questions from custom documents (PDF, TXT, DOCX), using Google Gemini, LangChain, LangGraph, and LangSmith.

## Features
- Multi-format document ingestion (PDF, TXT, DOCX)
- Semantic search via vector embeddings (ChromaDB)
- Conditional retrieval logic — rejects off-topic questions instead of hallucinating
- Source citation (file + page number) for every answer
- Full observability via LangSmith tracing

## Tech Stack
- Python
- LangChain / LangGraph
- Google Gemini (LLM + embeddings)
- ChromaDB (vector store)
- LangSmith (tracing/observability)

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


6. Ask a question:
python graph_query.py


## How It Works

1. Documents are loaded and split into chunks.
2. Chunks are embedded and stored in a vector database.
3. A user's question is embedded and matched against stored chunks.
4. If no relevant match is found, the system declines to answer rather than guessing.
5. Relevant chunks are passed to Gemini to generate a grounded, cited answer.