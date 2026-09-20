from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import TypedDict
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

load_dotenv()

# --- Your existing RAG setup, unchanged ---
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    found_relevant: bool

def retrieve(state: RAGState) -> dict:
    results = vectorstore.similarity_search_with_score(state["question"], k=3)
    RELEVANCE_THRESHOLD = 0.7
    relevant_docs = [doc for doc, score in results if score < RELEVANCE_THRESHOLD]
    if not relevant_docs:
        return {"found_relevant": False, "context": ""}
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    return {"found_relevant": True, "context": context}

def generate(state: RAGState) -> dict:
    prompt = f"""Answer using ONLY the context below. If it's not there, say you don't know.

Context:
{state['context']}

Question: {state['question']}
"""
    response = llm.invoke(prompt)
    return {"answer": response.text}

def no_answer_found(state: RAGState) -> dict:
    return {"answer": "I don't have relevant information in the document to answer that question."}

def route_after_retrieve(state: RAGState) -> str:
    return "generate" if state["found_relevant"] else "no_answer_found"

builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("no_answer_found", no_answer_found)
builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", route_after_retrieve)
builder.add_edge("generate", END)
builder.add_edge("no_answer_found", END)
graph = builder.compile()

# --- NEW: the web API wrapper ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    result = graph.invoke({"question": q.question})
    return {"answer": result["answer"]}