from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import openai
import pinecone
import time
import uuid

# Initialize FastAPI
app = FastAPI()

# OpenAI API Key
openai.api_key = "your_openai_api_key"

# Pinecone Initialization
pinecone.init(api_key="your_pinecone_api_key", environment="us-west1-gcp")  # Update the environment
index_name = "document-embeddings"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)  # Dimension for "text-embedding-ada-002"
index = pinecone.Index(index_name)

# Data Models
class Document(BaseModel):
    title: str
    content: str
    tags: list[str]

class Query(BaseModel):
    question: str
    top_k: int = 3  # Number of most relevant documents to retrieve
    document_filter: list[str] = None  # Titles, tags, or IDs to filter documents

class ProgressResponse(BaseModel):
    message: str
    progress: int

# Progress tracking (global dictionary, can be replaced by DB or task queue in production)
ingestion_progress = {}

@app.post("/ingest")
async def ingest_document(doc: Document, background_tasks: BackgroundTasks):
    """
    Ingest a document by generating its embedding and storing it in Pinecone.
    Tracks the progress of the ingestion.
    """
    try:
        # Create a unique task ID for this ingestion
        task_id = str(uuid.uuid4())
        ingestion_progress[task_id] = {"message": "Ingestion started", "progress": 0}

        # Start background task for document ingestion
        background_tasks.add_task(process_ingestion, doc, task_id)
        
        return {"message": "Document ingestion started. You will receive progress updates.", "task_id": task_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_ingestion(doc: Document, task_id: str):
    try:
        # Step 1: Simulate the initial step (embedding generation)
        ingestion_progress[task_id] = {"message": "Generating embeddings...", "progress": 20}
        time.sleep(2)  # Simulate time delay for embedding generation
        embedding_response = openai.Embedding.create(
            input=doc.content,
            model="text-embedding-ada-002"
        )
        embedding = embedding_response['data'][0]['embedding']
        ingestion_progress[task_id] = {"message": "Embeddings generated successfully.", "progress": 50}
        
        # Step 2: Simulate uploading embeddings to Pinecone
        ingestion_progress[task_id] = {"message": "Uploading embeddings to Pinecone...", "progress": 70}
        time.sleep(3)  # Simulate time delay for Pinecone upsert
        index.upsert(items=[{
            "id": doc.title,  # Unique ID for the document
            "values": embedding,
            "metadata": {"title": doc.title, "tags": doc.tags, "content": doc.content}
        }])
        ingestion_progress[task_id] = {"message": f"Ingestion of document '{doc.title}' completed successfully.", "progress": 100}
    
    except Exception as e:
        ingestion_progress[task_id] = {"message": f"Error during ingestion: {str(e)}", "progress": 0}

@app.get("/ingestion-progress/{task_id}")
async def ingestion_progress_status(task_id: str):
    """
    Get the current progress of a document ingestion task.
    """
    if task_id not in ingestion_progress:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    return ingestion_progress[task_id]

@app.post("/qa")
async def answer_question(query: Query):
    """
    Handles the Q&A process:
    1. Generates a question embedding.
    2. Filters documents based on user-specified criteria.
    3. Retrieves relevant documents using Pinecone.
    4. Generates an answer using OpenAI.
    """
    try:
        # Generate embedding for the question
        question_embedding_response = openai.Embedding.create(
            input=query.question,
            model="text-embedding-ada-002"
        )
        question_embedding = question_embedding_response['data'][0]['embedding']
        
        # Build filter query if document_filter is provided
        filter_query = {}
        if query.document_filter:
            filter_query = {
                "$or": [
                    {"title": {"$in": query.document_filter}},
                    {"tags": {"$in": query.document_filter}},
                ]
            }
        
        # Query Pinecone for relevant embeddings with optional filtering
        search_results = index.query(
            vector=question_embedding,
            top_k=query.top_k,
            include_metadata=True,
            filter=filter_query
        )
        
        # Prepare context from the top results
        contexts = [match['metadata']['content'] for match in search_results['matches']]
        context_string = "\n\n".join(contexts)
        
        # Generate answer using OpenAI
        chat_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Answer this question based on the following context: {context_string}\n\nQuestion: {query.question}"}
            ]
        )
        answer = chat_response['choices'][0]['message']['content']
        
        return {"answer": answer, "context": contexts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """
    List all documents stored in Pinecone.
    """
    try:
        # Retrieve metadata of all documents
        documents = index.describe_index_stats()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
