from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import tempfile
import os

def process_pdf(pdf_file, collection_name: str):
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    # Load and split
    loader = PyPDFLoader(file_path=tmp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    split_docs = text_splitter.split_documents(docs)

    # Embeddings and indexing
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://localhost:6333",  # Qdrant local instance
        collection_name=collection_name,
        embedding=embedding_model
    )

    os.remove(tmp_path)  # Cleanup temp file