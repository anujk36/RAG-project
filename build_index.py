from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

from app.config import settings
from app.logging_config import setup_logging
from app.image_processing import build_image_documents

load_dotenv()
setup_logging()

DOCUMENTS_DIR = Path(settings.documents_dir)

# 1. Load the text-based documents
pdf_loader = PyPDFDirectoryLoader(str(DOCUMENTS_DIR))
pdf_docs = pdf_loader.load()

txt_loader = DirectoryLoader(str(DOCUMENTS_DIR), glob="**/*.txt", loader_cls=TextLoader)
txt_docs = txt_loader.load()

docx_loader = DirectoryLoader(str(DOCUMENTS_DIR), glob="**/*.docx", loader_cls=Docx2txtLoader)
docx_docs = docx_loader.load()

text_documents = pdf_docs + txt_docs + docx_docs
print(f"Loaded {len(pdf_docs)} PDF pages, {len(txt_docs)} txt files, {len(docx_docs)} docx files")

# 2. Extract images embedded in PDFs + standalone image files, and caption
# each one with Gemini's vision model so they become searchable text
vision_llm = ChatGoogleGenerativeAI(model=settings.llm_model)
image_documents = build_image_documents(DOCUMENTS_DIR, vision_llm)
print(f"Captioned {len(image_documents)} images")

# 3. Chunk the text documents into smaller pieces (image captions are already
# short and self-contained, so they're indexed as single chunks)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
text_chunks = splitter.split_documents(text_documents)
print(f"Split into {len(text_chunks)} text chunks")

chunks = text_chunks + image_documents

# 4. Set up the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model=settings.embedding_model)

# 5. Embed and store chunks in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=settings.chroma_persist_dir
)

print(f"Done! Vector store saved to {settings.chroma_persist_dir} "
      f"({len(chunks)} total chunks: {len(text_chunks)} text + {len(image_documents)} image)")
