"""
Deep Research Models — Pydantic models and dataclasses for the research system.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime


# ═══════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════

class ResearchStatus(str, Enum):
    """State machine for research sessions"""
    IDLE = "idle"
    PLANNING = "planning"
    RESEARCHING = "researching"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    ERROR = "error"


class ResearchNodeType(str, Enum):
    """Node types in the Research Context Graph"""
    SESSION = "research_session"
    PLAN = "research_plan"
    STEP = "research_step"
    WEB_SOURCE = "web_source"
    FINDING = "finding"
    SYNTHESIS = "synthesis"
    SEARCH_QUERY = "search_query"


class ResearchEdgeType(str, Enum):
    """Edge types in the Research Context Graph"""
    HAS_PLAN = "has_plan"
    HAS_STEP = "has_step"
    EXECUTED_QUERY = "executed_query"
    FOUND_SOURCE = "found_source"
    EXTRACTED_FROM = "extracted_from"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SYNTHESIZED_FROM = "synthesized_from"
    DERIVED_FROM = "derived_from"
    BELONGS_TO = "belongs_to"


# ═══════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════

class ResearchConfig(BaseModel):
    """User-configurable research parameters"""
    max_sources: int = Field(default=20, ge=5, le=50, description="Maximum web sources to scrape")
    max_steps: int = Field(default=8, ge=2, le=15, description="Maximum research plan steps")
    depth_limit: int = Field(default=3, ge=1, le=5, description="RLM recursion depth limit")
    max_content_per_page: int = Field(default=15000, ge=5000, le=50000, description="Max chars to extract per page")
    scrape_timeout: float = Field(default=15.0, ge=5.0, le=60.0, description="Timeout per page scrape (seconds)")
    concurrent_scrapes: int = Field(default=5, ge=1, le=10, description="Max concurrent scrape operations")


# ═══════════════════════════════════════════════════
# Research Plan
# ═══════════════════════════════════════════════════

class ResearchStepDef(BaseModel):
    """A single step in the research plan"""
    step_index: int
    title: str
    description: str
    search_queries: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, running, completed, skipped


class ResearchPlan(BaseModel):
    """LLM-generated research plan"""
    steps: List[ResearchStepDef]
    strategy: str = "breadth-first"
    estimated_sources: int = 10


# ═══════════════════════════════════════════════════
# Scraped Content
# ═══════════════════════════════════════════════════

class ScrapedPage(BaseModel):
    """Result of scraping a single web page"""
    url: str
    title: str = ""
    content: str = ""
    word_count: int = 0
    domain: str = ""
    scraped_at: str = ""
    success: bool = False
    error: Optional[str] = None


# ═══════════════════════════════════════════════════
# Findings & Synthesis
# ═══════════════════════════════════════════════════

class Finding(BaseModel):
    """An extracted fact/insight from a web source"""
    id: str = ""
    content: str
    source_url: str = ""
    source_title: str = ""
    extraction_type: str = "fact"  # fact, statistic, quote, analysis, definition
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    depth: int = 0


class SourceInfo(BaseModel):
    """Source metadata for the final report"""
    url: str
    title: str
    domain: str
    relevance_score: float = 0.0
    findings_count: int = 0


# ═══════════════════════════════════════════════════
# Progress & Events
# ═══════════════════════════════════════════════════

class ResearchProgressEvent(BaseModel):
    """Real-time progress event sent to frontend"""
    event_type: str  # plan_generated, step_started, search_completed, source_scraped,
                     # findings_extracted, step_completed, synthesis_started, research_complete, error
    step_index: Optional[int] = None
    total_steps: Optional[int] = None
    sources_scraped: int = 0
    total_findings: int = 0
    current_activity: str = ""
    thinking: Optional[str] = None  # Intermediate LLM reasoning visible to user
    data: Optional[Dict] = None  # Additional event-specific data
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ═══════════════════════════════════════════════════
# Final Result
# ═══════════════════════════════════════════════════

class ResearchResult(BaseModel):
    """Final deep research output"""
    session_id: str
    query: str
    report: str = ""  # Full markdown report
    sources: List[SourceInfo] = Field(default_factory=list)
    total_sources_scraped: int = 0
    total_findings: int = 0
    research_time_ms: float = 0
    status: ResearchStatus = ResearchStatus.COMPLETE
    decision_trace: List[Dict] = Field(default_factory=list)  # Provenance chain
