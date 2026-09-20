from pathlib import Path
from typing import TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.exceptions import ModelRateLimitError
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END

from app.config import settings
from app.logging_config import logger
from app.exceptions import (
    DocumentRetrievalError,
    LLMGenerationError,
    EmptyQuestionError,
    IndexingError,
    RateLimitedError,
)
from app.image_processing import extract_images_from_pdf, build_image_documents


class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    found_relevant: bool
    images: list[str]


class RAGService:
    def __init__(self):
        logger.info("Initializing RAG service...")
        self.embeddings = GoogleGenerativeAIEmbeddings(model=settings.embedding_model)
        self.vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=self.embeddings
        )
        self.llm = ChatGoogleGenerativeAI(model=settings.llm_model)
        self.graph = self._build_graph()
        logger.info("RAG service ready.")

    def _retrieve(self, state: RAGState) -> dict:
        try:
            results = self.vectorstore.similarity_search_with_score(
                state["question"], k=settings.retrieval_k
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise DocumentRetrievalError(str(e))

        relevant_docs = [doc for doc, score in results if score < settings.relevance_threshold]

        if not relevant_docs:
            logger.info("No relevant documents found for question.")
            return {"found_relevant": False, "context": "", "images": []}

        context_parts = []
        images = []
        for doc in relevant_docs:
            if doc.metadata.get("type") == "image":
                context_parts.append(f"[Image: {doc.metadata.get('source')}] {doc.page_content}")
                images.append(doc.metadata.get("source"))
            else:
                context_parts.append(doc.page_content)

        context = "\n\n".join(context_parts)
        return {"found_relevant": True, "context": context, "images": images}

    def _generate(self, state: RAGState) -> dict:
        prompt = f"""Answer using ONLY the context below. If it's not there, say you don't know.

Context:
{state['context']}

Question: {state['question']}
"""
        try:
            response = self.llm.invoke(prompt)
        except ModelRateLimitError as e:
            logger.error(f"LLM rate-limited: {e}")
            raise RateLimitedError(
                "The AI model's rate limit or daily quota has been exceeded. Please try again later."
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMGenerationError(str(e))

        return {"answer": response.text}

    def _no_answer_found(self, state: RAGState) -> dict:
        return {"answer": "I don't have relevant information in the document to answer that question."}

    def _route_after_retrieve(self, state: RAGState) -> str:
        return "generate" if state["found_relevant"] else "no_answer_found"

    def _build_graph(self):
        builder = StateGraph(RAGState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_node("no_answer_found", self._no_answer_found)
        builder.add_edge(START, "retrieve")
        builder.add_conditional_edges("retrieve", self._route_after_retrieve)
        builder.add_edge("generate", END)
        builder.add_edge("no_answer_found", END)
        return builder.compile()

    def add_pdf(self, pdf_path: Path) -> dict:
        """Processes a single PDF (text + embedded images) and adds it to the
        already-open vector store, so it's searchable immediately."""
        pdf_path = Path(pdf_path)
        documents_dir = pdf_path.parent

        try:
            pages = PyPDFLoader(str(pdf_path)).load()
        except Exception as e:
            logger.error(f"Failed to load {pdf_path}: {e}")
            raise IndexingError(f"Could not read PDF: {e}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        text_chunks = splitter.split_documents(pages)

        extracted_images = extract_images_from_pdf(pdf_path, documents_dir)
        image_documents = build_image_documents(documents_dir, self.llm, images=extracted_images)

        chunks = text_chunks + image_documents
        if chunks:
            try:
                self.vectorstore.add_documents(chunks)
            except Exception as e:
                logger.error(f"Failed to add {pdf_path} to vector store: {e}")
                raise IndexingError(f"Could not index PDF: {e}")

        logger.info(f"Indexed {pdf_path.name}: {len(text_chunks)} text chunks, {len(image_documents)} images")
        return {"text_chunks": len(text_chunks), "images": len(image_documents)}

    def ask(self, question: str) -> dict:
        if not question or not question.strip():
            raise EmptyQuestionError("Question cannot be empty.")

        logger.info(f"Processing question: {question}")
        result = self.graph.invoke({"question": question})
        return {"answer": result["answer"], "images": result.get("images", [])}


rag_service = RAGService()
