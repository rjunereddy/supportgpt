from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os

from agents.orchestrator import app as agent_app
from retrieval.vector_store import (
    ingest_uploaded_files,
    get_index_stats,
    reset_index
)
from config.settings import get_settings, update_keys

app = FastAPI(
    title="SupportGPT API",
    description="Multi-Agent Retrieval-Augmented Customer Support with Document Upload and Dynamic Key Support",
    version="1.1.0"
)

class QueryRequest(BaseModel):
    query: str
    provider: Optional[str] = None  # "groq" or "openai"
    model: Optional[str] = None
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    documents_retrieved: int
    hallucination_detected: bool
    iterations: int
    sources: Optional[List[str]] = []

class KeyConfigRequest(BaseModel):
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    persist: bool = True

@app.get("/")
def health_check():
    settings = get_settings()
    has_groq = bool(settings.groq_api_key and settings.groq_api_key != "mock")
    has_openai = bool(settings.openai_api_key and settings.openai_api_key != "mock")
    return {
        "status": "healthy",
        "service": "SupportGPT",
        "groq_configured": has_groq,
        "openai_configured": has_openai
    }

@app.get("/stats")
def index_stats():
    """Returns current vector store stats and list of uploaded documents."""
    return get_index_stats()

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Uploads multiple documents (.txt, .md, .pdf, .csv, .json) and indexes them into the knowledge base."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for upload.")
    
    files_data = []
    for f in files:
        content = await f.read()
        files_data.append({
            "filename": f.filename,
            "content": content
        })
    
    try:
        result = ingest_uploaded_files(files_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process files: {str(e)}")

@app.post("/reset-index")
def reset_knowledge_base():
    """Resets the vector index back to default baseline."""
    return reset_index()

@app.post("/config/keys")
def configure_keys(request: KeyConfigRequest):
    """Updates API keys for Groq and OpenAI."""
    update_keys(
        openai_api_key=request.openai_api_key,
        groq_api_key=request.groq_api_key,
        persist=request.persist
    )
    settings = get_settings()
    return {
        "status": "success",
        "message": "API keys updated successfully.",
        "groq_configured": bool(settings.groq_api_key and settings.groq_api_key != "mock"),
        "openai_configured": bool(settings.openai_api_key and settings.openai_api_key != "mock")
    }

@app.post("/query", response_model=QueryResponse)
def query_support(request: QueryRequest):
    try:
        # Initialize state for LangGraph
        initial_state = {
            "query": request.query,
            "intent": "",
            "documents": [],
            "draft_answer": "",
            "is_hallucination": False,
            "iterations": 0,
            "provider": request.provider,
            "model": request.model,
            "groq_api_key": request.groq_api_key,
            "openai_api_key": request.openai_api_key
        }
        
        # Run the agent graph
        final_state = agent_app.invoke(initial_state)
        docs = final_state.get("documents", [])
        
        return QueryResponse(
            answer=final_state.get("draft_answer", "Sorry, I couldn't generate an answer."),
            documents_retrieved=len(docs),
            hallucination_detected=final_state.get("is_hallucination", False),
            iterations=final_state.get("iterations", 0),
            sources=docs[:5]  # Top source excerpts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
