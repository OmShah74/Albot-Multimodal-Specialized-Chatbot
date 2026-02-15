"""
Memory Models — Pydantic schemas for the agent-grade memory system.

Layers:
  L0 Working Memory  → in-memory dict (no model — ephemeral)
  L1 Session Memory   → ReasoningTrace, WebInteractionLog
  L2 Semantic Memory  → MemoryFragment  (cross-session, decay-pruned)
  L3 Procedural Memory→ MemoryFragment with type=SOLUTION
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class MemoryFragmentType(str, Enum):
    """Classification of extracted memory fragments."""
    KNOWLEDGE = "knowledge"       # Factual information
    SOLUTION = "solution"         # Problem → resolution pattern
    PREFERENCE = "preference"     # Detected user preference
    ENTITY = "entity"             # Named entity + surrounding context


# ──────────────────────────────────────────────
# Core Memory Models
# ──────────────────────────────────────────────

class MemoryFragment(BaseModel):
    """
    A discrete, reusable unit of memory extracted from a conversation turn.
    Stored in both SQLite (metadata) and ArangoDB (embedding + graph).
    """
    fragment_id: str
    session_id: str
    fragment_type: MemoryFragmentType
    content: str
    tags: List[str] = Field(default_factory=list)
    embedding_id: Optional[str] = None         # ArangoDB node _key
    namespace: str = "global"
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = 0
    created_at: str = ""
    last_accessed: Optional[str] = None
    is_deleted: bool = False


class ReasoningTrace(BaseModel):
    """
    Structured record of every step the system took to answer one user turn.
    Provides full explainability without leaking raw chain-of-thought.
    """
    trace_id: str
    session_id: str
    turn_index: int
    user_query: str
    reformulated_query: Optional[str] = None
    retrieved_doc_ids: List[str] = Field(default_factory=list)
    retrieved_doc_sources: List[str] = Field(default_factory=list)
    retrieval_scores: Dict[str, float] = Field(default_factory=dict)
    algorithms_used: List[str] = Field(default_factory=list)
    web_search_triggered: bool = False
    web_urls_searched: List[str] = Field(default_factory=list)
    web_snippets: List[Dict] = Field(default_factory=list)
    search_mode: str = "web_search"
    synthesis_model: str = ""
    answer_summary: str = ""
    total_time_ms: float = 0.0
    created_at: str = ""


class WebInteractionLog(BaseModel):
    """Single web search result record tied to a session turn."""
    session_id: str
    turn_index: int
    provider: str
    query_sent: str
    url: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    relevance_score: float = 0.0
    created_at: str = ""


class SessionMemoryConfig(BaseModel):
    """
    Per-session retrieval scope configuration.
    Controls which knowledge sources and namespaces are active.
    """
    session_id: str
    active_namespaces: List[str] = Field(default_factory=lambda: ["global"])
    source_filters: List[str] = Field(default_factory=list)    # empty = all sources
    include_web_history: bool = True
    include_fragments: bool = True
    updated_at: str = ""


class RetrievalScope(BaseModel):
    """
    Resolved retrieval scope — the output of NamespaceResolver.
    Passed into the retrieval engine to constrain searches.
    """
    source_filters: List[str] = Field(default_factory=list)
    active_namespaces: List[str] = Field(default_factory=lambda: ["global"])
    include_web_history: bool = True
    include_fragments: bool = True
