import streamlit as st
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv

# load_dotenv()
qdrant_api_key = st.secrets["QDRANT_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] = openai_api_key

client = OpenAI()

def get_vector_store(collection_name):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    return QdrantVectorStore.from_existing_collection(
        url="https://07b80d28-afc8-4779-998f-63c81d8a30ca.eu-west-2-0.aws.cloud.qdrant.io:6333",
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
    You are an assistant helping the user understand the content of a PDF file.

    Answer only using the context provided below, and guide the user to the correct page number when relevant.

    Context:
    {context}
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message.content