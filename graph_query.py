from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# 1. Define the shape of our "clipboard" (state)
class RAGState(TypedDict):
    question: str
    context: str
    answer: str

# 2. Define each node (worker station)
def retrieve(state: RAGState) -> dict:
    docs = vectorstore.similarity_search(state["question"], k=2)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"context": context}

def generate(state: RAGState) -> dict:
    prompt = f"""Answer using ONLY the context below. If it's not there, say you don't know.

Context:
{state['context']}

Question: {state['question']}
"""
    response = llm.invoke(prompt)
    return {"answer": response.text}

# 3. Build the graph: wire nodes together
builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

# 4. Run it
result = graph.invoke({"question": "How many liters per hour can the AquaBeam purify?"})
print(result["answer"])