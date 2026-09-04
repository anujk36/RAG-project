from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 1. Load the document
loader = TextLoader("data.txt")
documents = loader.load()

# 2. Chunk it into smaller pieces
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# 3. Set up the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 4. Embed and store chunks in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Done! Vector store saved to ./chroma_db")