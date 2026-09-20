from typing import TypedDict
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

from app.config import settings
from app.logging_config import logger
from app.exceptions import DocumentRetrievalError, LLMGenerationError, EmptyQuestionError


class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    found_relevant: bool


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
            return {"found_relevant": False, "context": ""}

        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        return {"found_relevant": True, "context": context}

    def _generate(self, state: RAGState) -> dict:
        prompt = f"""Answer using ONLY the context below. If it's not there, say you don't know.

Context:
{state['context']}

Question: {state['question']}
"""
        try:
            response = self.llm.invoke(prompt)
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

    def ask(self, question: str) -> str:
        if not question or not question.strip():
            raise EmptyQuestionError("Question cannot be empty.")

        logger.info(f"Processing question: {question}")
        result = self.graph.invoke({"question": question})
        return result["answer"]


rag_service = RAGService()
