# app/ingestion.py
import openai
import pinecone
import time
from app.models import Document
from fastapi import HTTPException
import uuid

# OpenAI API Key
openai.api_key = "your_openai_api_key"

# Pinecone Initialization
pinecone.init(api_key="your_pinecone_api_key", environment="us-west1-gcp")  # Update the environment
index_name = "document-embeddings"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)  # Dimension for "text-embedding-ada-002"
index = pinecone.Index(index_name)

# Global dictionary to track ingestion progress (for simplicity)
ingestion_progress = {}

def process_ingestion(doc: Document, task_id: str):
    try:
        # Step 1: Generate embeddings
        ingestion_progress[task_id] = {"message": "Generating embeddings...", "progress": 20}
        time.sleep(2)  # Simulate delay for embedding generation
        embedding_response = openai.Embedding.create(
            input=doc.content,
            model="text-embedding-ada-002"
        )
        embedding = embedding_response['data'][0]['embedding']
        ingestion_progress[task_id] = {"message": "Embeddings generated successfully.", "progress": 50}
        
        # Step 2: Upload embeddings to Pinecone
        ingestion_progress[task_id] = {"message": "Uploading embeddings to Pinecone...", "progress": 70}
        time.sleep(3)  # Simulate delay for Pinecone upsert
        index.upsert(items=[{
            "id": doc.title,  # Unique ID for the document
            "values": embedding,
            "metadata": {"title": doc.title, "tags": doc.tags, "content": doc.content}
        }])
        ingestion_progress[task_id] = {"message": f"Ingestion of document '{doc.title}' completed successfully.", "progress": 100}
    
    except Exception as e:
        ingestion_progress[task_id] = {"message": f"Error during ingestion: {str(e)}", "progress": 0}
        raise HTTPException(status_code=500, detail=str(e))

def get_ingestion_progress(task_id: str):
    if task_id not in ingestion_progress:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return ingestion_progress[task_id]
