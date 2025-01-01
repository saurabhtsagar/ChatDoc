# app/qa.py
import openai
import pinecone
from app.models import Query
from fastapi import HTTPException

# Pinecone Initialization
index_name = "document-embeddings"
index = pinecone.Index(index_name)

openai.api_key = "your_openai_api_key"

def answer_question(query: Query):
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
