# app/models.py
from pydantic import BaseModel
from typing import List, Optional

class Document(BaseModel):
    title: str
    content: str
    tags: List[str]

class Query(BaseModel):
    question: str
    top_k: int = 3  # Number of most relevant documents to retrieve
    document_filter: Optional[List[str]] = None  # Titles, tags, or IDs to filter documents

class ProgressResponse(BaseModel):
    message: str
    progress: int
