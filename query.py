from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

load_dotenv()

# Reconnect to the same embedding model and vector store we built earlier
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# The chat model that will generate the final answer
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# Ask a question
question = "How many liters per hour can the AquaBeam purify?"

# 5. RETRIEVE: find the most relevant chunks
retrieved_docs = vectorstore.similarity_search(question, k=2)

print("--- Retrieved chunks ---")
for doc in retrieved_docs:
    print(doc.page_content)
    print("---")

# Combine retrieved chunks into one block of context
context = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 6. GENERATE: ask the LLM to answer using ONLY that context
prompt = f"""Answer the question using ONLY the context below. 
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
"""

response = llm.invoke(prompt)
print("\n--- Answer ---")
print(response.text)