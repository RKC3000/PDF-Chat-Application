import os
import streamlit as st
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

qdrant_api_key = os.getenv("QDRANT_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key

client = OpenAI()

def get_vector_store(collection_name):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large") # This is the embedding model that will be used to create embeddings for the documents.
    return QdrantVectorStore.from_existing_collection(
        url="https://7d867feb-e640-48d8-bfcb-e992728e06c5.eu-west-2-0.aws.cloud.qdrant.io:6333",
        api_key=qdrant_api_key,
        collection_name=collection_name,
        embedding=embedding_model
    )

def ask_question(vector_db, query: str):
    results = vector_db.similarity_search(query)

    context = "\n\n".join(
        [f"Page Content: {r.page_content}\nPage Number: {r.metadata.get('page_label', '?')}" for r in results]
    )

    system_prompt = f"""
    You are an intelligent assistant helping a user chat with their PDF.

    Instructions:
    1. Primary Source: Answer the question using ONLY the provided Context from the PDF.
    2. Elaboration: You may briefly use general knowledge or common sense to explain concepts or fill logical gaps, provided it is directly relevant to the context.
    3. Uncertainty: If the answer cannot be derived from the context at all, simply state "I am not sure about the answer based on the provided document."

    Context:
    {context}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message.content
