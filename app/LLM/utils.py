# app/utils.py
import openai
import pinecone

def init_openai(api_key: str):
    openai.api_key = api_key

def init_pinecone(api_key: str, environment: str, index_name: str):
    pinecone.init(api_key=api_key, environment=environment)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)  # Dimension for "text-embedding-ada-002"
    return pinecone.Index(index_name)
