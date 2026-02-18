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
        "http://localhost:8010",
        "http://127.0.0.1:8010",
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
class RetrievalConfig(BaseModel):
    """Configuration for retrieval algorithms"""
    mode: str = "advanced"  # "fast" or "advanced"
    use_vector: bool = True
    use_graph: bool = True
    use_bm25: bool = True
    use_pagerank: bool = True
    use_structural: bool = True
    use_mmr: bool = True


class QueryRequest(BaseModel):
    query: str
    chat_id: str = "default"  # Added chat_id with default for backward compatibility
    modalities: Optional[List[str]] = None
    retrieval_config: Optional[RetrievalConfig] = None
    search_mode: str = "web_search"  # "web_search" | "knowledge_base"


class QueryMetrics(BaseModel):
    """Query performance metrics"""
    total_time_ms: float = 0
    vector_time_ms: float = 0
    graph_time_ms: float = 0
    bm25_time_ms: float = 0
    synthesis_time_ms: float = 0
    results_count: int = 0
    mode: str = "advanced"
    algorithms_used: List[str] = []
    web_search_used: bool = False
    web_search_time_ms: float = 0
    web_providers_used: dict = {}
    search_mode: str = "web_search"


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    metrics: Optional[QueryMetrics] = None
    chat_title: Optional[str] = None



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
    """Process query with configurable retrieval"""
    try:
        from backend.models.config import Modality
        
        # Convert modality strings
        modalities = None
        if request.modalities:
            modalities = [Modality(m) for m in request.modalities]
        
        # Build retrieval config dict
        retrieval_config = None
        if request.retrieval_config:
            retrieval_config = request.retrieval_config.dict()
        
        # Query with config and search mode
        result = await rag_system.query(
            request.query, 
            request.chat_id, 
            modalities, 
            retrieval_config,
            search_mode=request.search_mode
        )
        
        # Build metrics if available
        metrics = None
        if 'metrics' in result:
            metrics = QueryMetrics(**result['metrics'])
        
        return QueryResponse(
            answer=result['answer'], 
            sources=result['sources'],
            metrics=metrics,
            chat_title=result.get('chat_title')
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CancelRequest(BaseModel):
    chat_id: str

@app.post("/query/cancel")
async def cancel_query(request: CancelRequest):
    """Cancel an in-progress query for a given chat"""
    try:
        rag_system.cancel_query(request.chat_id)
        return {"status": "cancelled", "chat_id": request.chat_id}
    except Exception as e:
        logger.error(f"Cancel error: {e}")
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



# --- Chat Management Endpoints ---

class CreateChatRequest(BaseModel):
    title: str = "New Chat"

class RenameChatRequest(BaseModel):
    title: str

@app.get("/chats")
async def get_chats():
    """Get all chat sessions"""
    try:
        if not rag_system:
            return []
        return rag_system.get_chats()
    except Exception as e:
        logger.error(f"Failed to get chats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chats")
async def create_chat(request: CreateChatRequest):
    """Create a new chat"""
    try:
        return rag_system.create_chat(request.title)
    except Exception as e:
        logger.error(f"Failed to create chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Get specific chat"""
    try:
        chat = rag_system.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        return chat
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/chats/{chat_id}")
async def rename_chat(chat_id: str, request: RenameChatRequest):
    """Rename a chat"""
    try:
        rag_system.rename_chat(chat_id, request.title)
        return {"status": "success", "id": chat_id, "title": request.title}
    except Exception as e:
        logger.error(f"Failed to rename chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat session"""
    try:
        rag_system.delete_chat(chat_id)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to delete chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/{chat_id}/history")
async def get_chat_history(chat_id: str):
    """Get history for specific chat"""
    try:
        if not rag_system:
            return []
        return rag_system.get_chat_history(chat_id)
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chats/{chat_id}/history")
async def clear_chat_history(chat_id: str):
    """Clear history for specific chat"""
    try:
        rag_system.clear_chat_history(chat_id)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Deprecated/Legacy Endpoints (mapped to default or removed if preferred)
# Keeping global /history compatible implies a default chat or aggregating all?
# For now, let's point /history to a default fallback or warn.
# Since we are moving to multi-chat, we should encourage using /chats/{id}/history.


# --- Memory System Endpoints ---

class SessionMemoryConfigRequest(BaseModel):
    active_namespaces: List[str] = ["global"]
    source_filters: List[str] = []
    include_web_history: bool = True
    include_fragments: bool = True

class CombineSessionsRequest(BaseModel):
    source_session_ids: List[str]
    target_namespace: str
    selected_fragment_ids: Optional[List[str]] = None

@app.get("/memory/{chat_id}/config")
async def get_memory_config(chat_id: str):
    """Get memory configuration for a chat session"""
    try:
        config = rag_system.memory.get_session_config(chat_id)
        return config.model_dump() if config else {}
    except Exception as e:
        logger.error(f"Failed to get memory config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/memory/{chat_id}/config")
async def set_memory_config(chat_id: str, request: SessionMemoryConfigRequest):
    """Update memory configuration for a chat session"""
    try:
        from backend.models.memory import SessionMemoryConfig
        from datetime import datetime
        config = SessionMemoryConfig(
            session_id=chat_id,
            active_namespaces=request.active_namespaces,
            source_filters=request.source_filters,
            include_web_history=request.include_web_history,
            include_fragments=request.include_fragments,
            updated_at=datetime.utcnow().isoformat()
        )
        rag_system.memory.set_session_config(config)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to set memory config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{chat_id}/traces")
async def get_reasoning_traces(chat_id: str):
    """Get all reasoning traces for a chat session"""
    try:
        traces = rag_system.memory.get_traces(chat_id)
        return {"traces": traces}
    except Exception as e:
        logger.error(f"Failed to get reasoning traces: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{chat_id}/fragments")
async def get_session_fragments(chat_id: str):
    """Get all memory fragments for a chat session"""
    try:
        fragments = rag_system.memory.get_session_fragments(chat_id)
        return {"fragments": fragments}
    except Exception as e:
        logger.error(f"Failed to get session fragments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{chat_id}/web-history")
async def get_web_history(chat_id: str):
    """Get web interaction logs for a chat session"""
    try:
        logs = rag_system.memory.get_web_history(chat_id)
        return {"web_interactions": logs}
    except Exception as e:
        logger.error(f"Failed to get web history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{chat_id}/dump")
async def get_memory_dump(chat_id: str):
    """Get complete structured memory dump for a chat session (conversation, fragments, web logs, traces, stats)"""
    try:
        dump = rag_system.memory.get_full_memory_dump(chat_id)
        return dump
    except Exception as e:
        logger.error(f"Failed to get memory dump: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/fragments/{fragment_id}")
async def delete_memory_fragment(fragment_id: str):
    """Delete a specific memory fragment"""
    try:
        rag_system.memory.delete_fragment(fragment_id)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to delete memory fragment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/web/{log_id}")
async def delete_web_log(log_id: int):
    """Delete a specific web interaction log"""
    try:
        rag_system.memory.delete_web_log(log_id)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to delete web log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/trace/{trace_id}")
async def delete_reasoning_trace(trace_id: str):
    """Delete a specific reasoning trace"""
    try:
        rag_system.memory.delete_reasoning_trace(trace_id)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to delete reasoning trace: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/message/{message_id}")
async def delete_chat_message(message_id: int):
    """Delete a specific chat message"""
    try:
        rag_system.memory.delete_message(message_id)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to delete chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/combine")
async def combine_sessions(request: CombineSessionsRequest):
    """Combine memories from multiple sessions into a target namespace"""
    try:
        count = rag_system.memory.combine_sessions(
            source_session_ids=request.source_session_ids,
            target_namespace=request.target_namespace,
            user_selected_fragment_ids=request.selected_fragment_ids
        )
        return {"status": "success", "fragments_copied": count}
    except Exception as e:
        logger.error(f"Failed to combine sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/namespaces")
async def get_namespaces(session_id: Optional[str] = None):
    """Get available memory namespaces and fragment counts"""
    try:
        namespaces = rag_system.namespace_resolver.get_available_namespaces(session_id)
        return {"namespaces": namespaces}
    except Exception as e:
        logger.error(f"Failed to get namespaces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════
# Deep Research Endpoints
# ═══════════════════════════════════════════════════

import asyncio
import uuid as _uuid
from datetime import datetime as _datetime

from backend.core.deep_research.models import (
    ResearchConfig, ResearchStatus, ResearchProgressEvent, ResearchResult
)
from backend.core.deep_research.deep_research_orchestrator import DeepResearchOrchestrator


# Deep research orchestrator (lazily initialized from rag_system)
_deep_research: Optional[DeepResearchOrchestrator] = None
_research_tasks: dict = {}  # session_id → asyncio.Task


def _get_deep_research() -> DeepResearchOrchestrator:
    """Get or create the deep research orchestrator from the RAG system."""
    global _deep_research
    if _deep_research is None:
        if not rag_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        _deep_research = DeepResearchOrchestrator(
            llm_router=rag_system.llm_router,
            web_search_engine=rag_system.web_search,
        )
    return _deep_research


class DeepResearchRequest(BaseModel):
    query: str
    chat_id: str = "default"
    config: Optional[ResearchConfig] = None


class DeepResearchStatusResponse(BaseModel):
    session_id: str
    status: str
    sources_scraped: int = 0
    total_findings: int = 0
    progress_events: List[dict] = []
    plan: Optional[dict] = None


@app.post("/research/start")
async def start_research(request: DeepResearchRequest):
    """Start a deep research session. Returns immediately with session_id."""
    try:
        orchestrator = _get_deep_research()
        session_id = str(_uuid.uuid4())
        config = request.config or ResearchConfig()
        
        # Persist the session
        if rag_system and hasattr(rag_system, 'chat_storage'):
            # First, save the user message to history so it persists on reload
            rag_system.chat_storage.save_chat_message(
                request.chat_id,
                "user",
                request.query
            )
            
            # Then create the research session
            rag_system.chat_storage.create_research_session({
                "id": session_id,
                "chat_id": request.chat_id,
                "query": request.query,
                "status": "planning",
                "config": config.model_dump(),
            })
        
        # Run research in background
        async def _run():
            try:
                async def on_progress(event: ResearchProgressEvent):
                    # Save progress to DB for polling
                    if rag_system and hasattr(rag_system, 'chat_storage'):
                        rag_system.chat_storage.save_research_progress(
                            session_id, event.model_dump()
                        )
                
                result = await orchestrator.run_research(
                    query=request.query,
                    session_id=session_id,
                    config=config,
                    on_progress=on_progress,
                )
                
                # Persist final result
                if rag_system and hasattr(rag_system, 'chat_storage'):
                    rag_system.chat_storage.update_research_session(session_id, {
                        "status": result.status.value,
                        "report": result.report,
                        "sources_scraped": result.total_sources_scraped,
                        "findings_count": result.total_findings,
                        "total_time_ms": result.research_time_ms,
                        "completed_at": _datetime.utcnow().isoformat(),
                    })
                    
                    # Also save the report as an assistant message in the chat
                    if result.report and request.chat_id:
                        rag_system.chat_storage.save_chat_message(
                            request.chat_id,
                            "assistant",
                            result.report,
                            sources=[s.url for s in result.sources],
                            metrics={
                                "type": "deep_research", 
                                "session_id": session_id,
                                "total_time_ms": result.research_time_ms,
                                "total_sources": result.total_sources_scraped,
                                "total_findings": result.total_findings
                            }
                        )
                    
                    # ── Memory post-processing for deep research ──
                    try:
                        _mem_uuid = __import__('uuid')
                        turn_index = rag_system.chat_storage.get_turn_count(request.chat_id)
                        
                        # 1. Save reasoning trace
                        rag_system.chat_storage.save_reasoning_trace({
                            "trace_id": str(_mem_uuid.uuid4()),
                            "session_id": request.chat_id,
                            "turn_index": turn_index,
                            "user_query": request.query,
                            "reformulated_query": None,
                            "retrieved_doc_ids": [],
                            "retrieved_doc_sources": [s.url for s in result.sources],
                            "retrieval_scores": {},
                            "algorithms_used": ["deep_research"],
                            "web_search_triggered": True,
                            "web_urls_searched": [s.url for s in result.sources],
                            "web_snippets": [],
                            "search_mode": "deep_research",
                            "synthesis_model": "",
                            "answer_summary": (result.report or "")[:300],
                            "total_time_ms": result.research_time_ms,
                            "created_at": _datetime.utcnow().isoformat(),
                        })
                        
                        # 2. Save web interaction logs for each source
                        web_logs = []
                        for src in result.sources:
                            web_logs.append({
                                "session_id": request.chat_id,
                                "turn_index": turn_index,
                                "provider": "deep_research",
                                "query_sent": request.query,
                                "url": src.url,
                                "title": src.title,
                                "snippet": "",
                                "relevance_score": src.relevance_score,
                            })
                        if web_logs:
                            rag_system.chat_storage.save_web_interaction_logs(web_logs)
                        
                        # 3. Save a memory fragment with research summary
                        rag_system.chat_storage.save_memory_fragment({
                            "fragment_id": str(_mem_uuid.uuid4()),
                            "session_id": request.chat_id,
                            "fragment_type": "deep_research_summary",
                            "content": f"Deep Research: {request.query}\n\n{(result.report or '')[:1000]}",
                            "tags": ["deep_research", session_id],
                            "namespace": "global",
                            "importance_score": 0.8,
                            "access_count": 0,
                            "created_at": _datetime.utcnow().isoformat(),
                        })
                        
                        logger.info(f"Deep research memory saved for chat {request.chat_id}: {len(web_logs)} web logs, 1 trace, 1 fragment")
                    except Exception as mem_err:
                        logger.warning(f"Deep research memory post-processing failed (non-fatal): {mem_err}")
                
            except Exception as e:
                logger.error(f"Background research failed: {e}")
                if rag_system and hasattr(rag_system, 'chat_storage'):
                    rag_system.chat_storage.update_research_session(session_id, {
                        "status": "error",
                        "completed_at": _datetime.utcnow().isoformat(),
                    })
        
        task = asyncio.create_task(_run())
        _research_tasks[session_id] = task
        
        return {
            "session_id": session_id,
            "status": "planning",
            "message": "Deep research started. Poll /research/{session_id}/status for updates.",
        }
        
    except Exception as e:
        logger.error(f"Failed to start research: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/research/{session_id}/status")
async def get_research_status(session_id: str):
    """Get the current status and progress events of a research session."""
    try:
        orchestrator = _get_deep_research()
        
        # Check in-memory state first (active session)
        session = orchestrator.get_session_status(session_id)
        if session:
            return DeepResearchStatusResponse(
                session_id=session_id,
                status=session.get("status", ResearchStatus.IDLE).value
                    if isinstance(session.get("status"), ResearchStatus)
                    else str(session.get("status", "idle")),
                sources_scraped=session.get("sources_scraped", 0),
                total_findings=session.get("total_findings", 0),
                progress_events=session.get("progress_events", []),
                plan=session.get("plan"),
            ).model_dump()
        
        # Fall back to DB
        if rag_system and hasattr(rag_system, 'chat_storage'):
            db_session = rag_system.chat_storage.get_research_session(session_id)
            if db_session:
                progress = rag_system.chat_storage.get_research_progress(session_id)
                return DeepResearchStatusResponse(
                    session_id=session_id,
                    status=db_session.get("status", "idle"),
                    sources_scraped=db_session.get("sources_scraped", 0),
                    total_findings=db_session.get("findings_count", 0),
                    progress_events=[p.get("data", p) for p in progress],
                    plan=db_session.get("plan"),
                ).model_dump()
        
        raise HTTPException(status_code=404, detail="Research session not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get research status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/{session_id}/stop")
async def stop_research(session_id: str):
    """Stop an in-progress research session. Triggers early synthesis."""
    try:
        orchestrator = _get_deep_research()
        orchestrator.cancel(session_id)
        
        return {
            "status": "cancelling",
            "message": "Research will stop after the current step and synthesize available findings.",
        }
        
    except Exception as e:
        logger.error(f"Failed to stop research: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/research/{session_id}/result")
async def get_research_result(session_id: str):
    """Get the final research report and sources."""
    try:
        orchestrator = _get_deep_research()
        
        # Check in-memory
        session = orchestrator.get_session_status(session_id)
        if session and session.get("report"):
            sources = session.get("sources", [])
            return {
                "session_id": session_id,
                "query": session.get("query", ""),
                "report": session.get("report", ""),
                "sources": sources,
                "status": session.get("status", ResearchStatus.IDLE).value
                    if isinstance(session.get("status"), ResearchStatus)
                    else str(session.get("status", "idle")),
                "sources_scraped": session.get("sources_scraped", 0),
                "total_findings": session.get("total_findings", 0),
            }
        
        # Fall back to DB
        if rag_system and hasattr(rag_system, 'chat_storage'):
            db_session = rag_system.chat_storage.get_research_session(session_id)
            if db_session:
                return {
                    "session_id": session_id,
                    "query": db_session.get("query", ""),
                    "report": db_session.get("report", ""),
                    "sources": [],
                    "status": db_session.get("status", "idle"),
                    "sources_scraped": db_session.get("sources_scraped", 0),
                    "total_findings": db_session.get("findings_count", 0),
                }
        
        raise HTTPException(status_code=404, detail="Research session not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get research result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/research/{session_id}/stream")
async def stream_research_progress(session_id: str):
    """SSE stream of research progress events."""
    from starlette.responses import StreamingResponse

    orchestrator = _get_deep_research()

    async def event_generator():
        last_idx = 0
        while True:
            session = orchestrator.get_session_status(session_id)
            if not session:
                yield f"data: {{\"event_type\": \"error\", \"current_activity\": \"Session not found\"}}\n\n"
                break

            events = session.get("progress_events", [])
            # Send new events since last check
            for event in events[last_idx:]:
                import json as _json
                yield f"data: {_json.dumps(event)}\n\n"
            last_idx = len(events)

            status = session.get("status")
            status_val = status.value if isinstance(status, ResearchStatus) else str(status)
            if status_val in ("complete", "cancelled", "error"):
                yield f"data: {{\"event_type\": \"done\", \"status\": \"{status_val}\"}}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
