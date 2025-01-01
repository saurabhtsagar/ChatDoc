# app/main.py
from fastapi import FastAPI, BackgroundTasks
from app.models import Document, Query
from app.ingestion import process_ingestion, get_ingestion_progress
from app.qa import answer_question
import uuid

app = FastAPI()

@app.post("/ingest")
async def ingest_document(doc: Document, background_tasks: BackgroundTasks):
    """
    Ingest a document by generating its embedding and storing it in Pinecone.
    Tracks the progress of the ingestion.
    """
    try:
        # Create a unique task ID for this ingestion
        task_id = str(uuid.uuid4())

        # Start background task for document ingestion
        background_tasks.add_task(process_ingestion, doc, task_id)
        
        return {"message": "Document ingestion started. You will receive progress updates.", "task_id": task_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ingestion-progress/{task_id}")
async def ingestion_progress_status(task_id: str):
    """
    Get the current progress of a document ingestion task.
    """
    return get_ingestion_progress(task_id)

@app.post("/qa")
async def answer_question_endpoint(query: Query):
    """
    Handles the Q&A process:
    1. Generates a question embedding.
    2. Filters documents based on user-specified criteria.
    3. Retrieves relevant documents using Pinecone.
    4. Generates an answer using OpenAI.
    """
    return answer_question(query)
