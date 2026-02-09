"""FastAPI backend for RAG system"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
from pathlib import Path
from loguru import logger

from backend.core.orchestrator import RAGOrchestrator


app = FastAPI(title="Albot API")

# CORS
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*" # Keep wildcard for now but credential issue might persist if * is used with credentials
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system: Optional[RAGOrchestrator] = None

@app.on_event("startup")
async def startup():
    global rag_system
    logger.info("Starting RAG system...")
    rag_system = RAGOrchestrator()


@app.on_event("shutdown")
async def shutdown():
    if rag_system:
        rag_system.shutdown()


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    modalities: Optional[List[str]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []


class APIKeyRequest(BaseModel):
    provider: str
    name: str
    key: str
    model_name: Optional[str] = None


class URLIngestRequest(BaseModel):
    url: str


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/sources")
async def list_sources():
    """List all unique sources"""
    try:
        return rag_system.list_sources()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sources/{source_name}")
async def delete_source(source_name: str):
    """Delete a document by source name"""
    try:
        rag_system.delete_document(source_name)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/system/reset")
async def reset_system():
    """Clear entire knowledge base"""
    try:
        rag_system.reset_system()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest uploaded file"""
    try:
        # Save file
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest
        result = rag_system.ingest_file(str(file_path))
        
        return result
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/url")
async def ingest_url(request: URLIngestRequest):
    """Ingest URL content"""
    try:
        result = rag_system.ingest_url(request.url)
        return result
    except Exception as e:
        logger.error(f"URL Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process query"""
    try:
        from backend.models.config import Modality
        
        # Convert modality strings
        modalities = None
        if request.modalities:
            modalities = [Modality(m) for m in request.modalities]
        
        # Query
        result = rag_system.query(request.query, modalities)
        
        return QueryResponse(answer=result['answer'], sources=result['sources'])
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api-keys/add")
async def add_api_key(request: APIKeyRequest):
    """Add API key"""
    try:
        rag_system.add_api_key(
            provider=request.provider,
            name=request.name,
            key=request.key,
            model_name=request.model_name
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api-keys/list")
async def list_api_keys():
    """List all API keys"""
    try:
        return rag_system.get_api_keys()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api-keys/{provider}/{name}")
async def delete_api_key(provider: str, name: str):
    """Delete an API key"""
    try:
        rag_system.delete_api_key(provider, name)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics")
async def get_statistics():
    """Get system statistics"""
    try:
        return rag_system.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)