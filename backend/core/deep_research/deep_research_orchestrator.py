"""
Deep Research Orchestrator — Main agentic research loop.
Ties together the Context Graph, RLM Engine, Web Scraper, and Research Planner
into a cancellable, progress-reporting deep research pipeline.

State machine: IDLE → PLANNING → RESEARCHING → SYNTHESIZING → COMPLETE
                                                    ↑              ↓
                                               CANCELLED ──────────┘
"""

import asyncio
import time
import uuid
from typing import List, Dict, Optional, Callable, Awaitable
from datetime import datetime
from loguru import logger

from backend.core.deep_research.models import (
    ResearchConfig, ResearchStatus, ResearchProgressEvent,
    ResearchResult, ResearchPlan, SourceInfo,
)
from backend.core.deep_research.context_graph import ResearchContextGraph
from backend.core.deep_research.rlm_engine import RecursiveResearchAgent
from backend.core.deep_research.web_scraper import DeepWebScraper
from backend.core.deep_research.research_planner import ResearchPlanner
from backend.core.deep_research.models import ResearchNodeType, ResearchEdgeType


# Type alias for progress callback
ProgressCallback = Callable[[ResearchProgressEvent], Awaitable[None]]


class DeepResearchOrchestrator:
    """
    Main orchestrator for deep research sessions.
    
    Flow:
    1. PLANNING: Generate a research plan using LLM
    2. RESEARCHING: For each step — search web, scrape URLs, extract findings
    3. SYNTHESIZING: Recursive synthesis of all findings into a report
    4. COMPLETE: Return the research report with sources and provenance
    
    Features:
    - Cancellable at any point between steps
    - Real-time progress events via callback
    - Configurable limits (max sources, max steps, depth limit)
    - Full provenance tracking via Context Graph
    """

    def __init__(
        self,
        llm_router,
        web_search_engine,
        config: Optional[ResearchConfig] = None,
    ):
        self.llm_router = llm_router
        self.web_search = web_search_engine
        self.config = config or ResearchConfig()
        
        # Active research sessions
        self._sessions: Dict[str, Dict] = {}
        self._cancelled: set = set()

    def cancel(self, session_id: str):
        """Mark a session as cancelled. The research loop will stop at the next step."""
        self._cancelled.add(session_id)
        if session_id in self._sessions:
            self._sessions[session_id]["status"] = ResearchStatus.CANCELLED
        logger.info(f"[DeepResearch] Session {session_id} marked for cancellation")

    def is_cancelled(self, session_id: str) -> bool:
        """Check if a session has been cancelled."""
        return session_id in self._cancelled

    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get the current status of a research session."""
        return self._sessions.get(session_id)

    async def run_research(
        self,
        query: str,
        session_id: Optional[str] = None,
        config: Optional[ResearchConfig] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ResearchResult:
        """
        Main research execution pipeline.
        
        Args:
            query: The research question
            session_id: Optional session ID (generated if not provided)
            config: Optional override for research configuration
            on_progress: Async callback for progress events
            
        Returns:
            ResearchResult with the full report, sources, and provenance
        """
        session_id = session_id or str(uuid.uuid4())
        config = config or self.config
        start_time = time.time()
        
        # Initialize components
        context_graph = ResearchContextGraph(session_id)
        planner = ResearchPlanner(self.llm_router, config)
        scraper = DeepWebScraper(config)
        rlm = RecursiveResearchAgent(self.llm_router, context_graph, config.depth_limit)
        
        # Track session state
        self._sessions[session_id] = {
            "session_id": session_id,
            "query": query,
            "status": ResearchStatus.IDLE,
            "config": config.model_dump(),
            "progress_events": [],
            "sources_scraped": 0,
            "total_findings": 0,
            "plan": None,
            "graph": context_graph,
        }
        
        async def emit(event: ResearchProgressEvent):
            """Emit a progress event and store it."""
            self._sessions[session_id]["progress_events"].append(event.model_dump())
            self._sessions[session_id]["sources_scraped"] = event.sources_scraped
            self._sessions[session_id]["total_findings"] = event.total_findings
            if on_progress:
                try:
                    await on_progress(event)
                except Exception as e:
                    logger.warning(f"[DeepResearch] Progress callback error: {e}")

        try:
            # ── Phase 1: PLANNING ──────────────────────────
            self._sessions[session_id]["status"] = ResearchStatus.PLANNING
            
            await emit(ResearchProgressEvent(
                event_type="status_update",
                current_activity="Generating research plan...",
                thinking="Analyzing the query and creating a structured research plan with targeted search queries.",
            ))
            
            plan = await planner.generate_plan(query)
            self._sessions[session_id]["plan"] = plan.model_dump()
            
            # Add plan to context graph
            session_node_id = context_graph.add_node(
                ResearchNodeType.SESSION,
                {"query": query, "status": "active"}
            )
            plan_node_id = context_graph.add_node(
                ResearchNodeType.PLAN,
                {
                    "strategy": plan.strategy,
                    "step_count": len(plan.steps),
                    "estimated_sources": plan.estimated_sources,
                }
            )
            context_graph.add_edge(session_node_id, plan_node_id, ResearchEdgeType.HAS_PLAN)
            
            await emit(ResearchProgressEvent(
                event_type="plan_generated",
                total_steps=len(plan.steps),
                current_activity=f"Research plan created with {len(plan.steps)} steps",
                data={
                    "steps": [{"index": s.step_index, "title": s.title} for s in plan.steps]
                },
            ))
            
            if self.is_cancelled(session_id):
                return await self._finalize(session_id, query, context_graph, rlm, [], start_time, cancelled=True)
            
            # ── Phase 2: RESEARCHING ──────────────────────
            self._sessions[session_id]["status"] = ResearchStatus.RESEARCHING
            
            total_sources_scraped = 0
            total_findings = 0
            step_syntheses = []
            
            for step in plan.steps:
                if self.is_cancelled(session_id):
                    break
                
                if total_sources_scraped >= config.max_sources:
                    logger.info(f"[DeepResearch] Reached max sources ({config.max_sources}), stopping")
                    break
                
                step_idx = step.step_index
                
                # Add step to graph
                step_node_id = context_graph.add_node(
                    ResearchNodeType.STEP,
                    {
                        "step_index": step_idx,
                        "title": step.title,
                        "description": step.description,
                        "status": "running",
                    }
                )
                context_graph.add_edge(plan_node_id, step_node_id, ResearchEdgeType.HAS_STEP)
                
                await emit(ResearchProgressEvent(
                    event_type="step_started",
                    step_index=step_idx,
                    total_steps=len(plan.steps),
                    sources_scraped=total_sources_scraped,
                    total_findings=total_findings,
                    current_activity=f"Step {step_idx + 1}/{len(plan.steps)}: {step.title}",
                    thinking=f"Researching: {step.description}",
                ))
                
                # ── 2a. Search the web ──
                all_urls = []
                for search_query in step.search_queries:
                    if self.is_cancelled(session_id):
                        break
                    
                    # Add search query to graph
                    query_node_id = context_graph.add_node(
                        ResearchNodeType.SEARCH_QUERY,
                        {"query_text": search_query, "depth": 0}
                    )
                    context_graph.add_edge(step_node_id, query_node_id, ResearchEdgeType.EXECUTED_QUERY)
                    
                    try:
                        search_results, _ = await self.web_search.search(search_query, top_k=5)
                        
                        urls = [r.url for r in search_results if r.url]
                        all_urls.extend(urls)
                        
                        # Link sources in graph
                        for r in search_results:
                            if r.url:
                                context_graph.add_edge(
                                    query_node_id,
                                    context_graph.add_node(
                                        ResearchNodeType.WEB_SOURCE,
                                        {"url": r.url, "title": r.title, "domain": getattr(r, 'domain', ''), "snippet": r.snippet[:200] if r.snippet else ""}
                                    ),
                                    ResearchEdgeType.FOUND_SOURCE
                                )
                        
                        await emit(ResearchProgressEvent(
                            event_type="search_completed",
                            step_index=step_idx,
                            total_steps=len(plan.steps),
                            sources_scraped=total_sources_scraped,
                            total_findings=total_findings,
                            current_activity=f"Found {len(urls)} results for: {search_query[:60]}",
                        ))
                        
                    except Exception as e:
                        logger.warning(f"[DeepResearch] Search failed for '{search_query}': {e}")
                
                if self.is_cancelled(session_id):
                    break
                
                # ── 2b. Scrape top URLs ──
                # Limit per step to spread sources across steps
                remaining_budget = config.max_sources - total_sources_scraped
                urls_to_scrape = list(dict.fromkeys(all_urls))[:min(5, remaining_budget)]
                
                if not urls_to_scrape:
                    context_graph.update_node(step_node_id, {"status": "completed", "findings_count": 0})
                    step_syntheses.append({"title": step.title, "synthesis": "No sources could be found for this step."})
                    continue
                
                scraped_pages = await scraper.scrape_urls(urls_to_scrape)
                
                # ── 2c. RLM extraction from each scraped page ──
                step_findings_count = 0
                
                for page in scraped_pages:
                    if self.is_cancelled(session_id):
                        break
                    
                    if not page.success or not page.content:
                        continue
                    
                    total_sources_scraped += 1
                    
                    # Find or create source node
                    source_node_id = context_graph.add_source(
                        url=page.url,
                        title=page.title,
                        domain=page.domain,
                        content_length=len(page.content),
                    )
                    
                    await emit(ResearchProgressEvent(
                        event_type="source_scraped",
                        step_index=step_idx,
                        total_steps=len(plan.steps),
                        sources_scraped=total_sources_scraped,
                        total_findings=total_findings,
                        current_activity=f"Scraped: {page.title[:60] or page.domain}",
                        data={"url": page.url, "title": page.title, "domain": page.domain},
                    ))
                    
                    # Extract findings using RLM
                    await emit(ResearchProgressEvent(
                        event_type="analyzing_source",
                        step_index=step_idx,
                        total_steps=len(plan.steps),
                        sources_scraped=total_sources_scraped,
                        total_findings=total_findings,
                        current_activity=f"Analyzing content from {page.domain}...",
                        thinking=f"Extracting key findings from {page.title or page.url}: {page.content[:150]}...",
                    ))
                    
                    findings = await rlm.extract_findings(
                        page_content=page.content,
                        research_query=query,
                        source_id=source_node_id,
                        source_url=page.url,
                        source_title=page.title,
                        step_node_id=step_node_id,
                    )
                    
                    step_findings_count += len(findings)
                    total_findings += len(findings)
                    
                    if findings:
                        await emit(ResearchProgressEvent(
                            event_type="findings_extracted",
                            step_index=step_idx,
                            total_steps=len(plan.steps),
                            sources_scraped=total_sources_scraped,
                            total_findings=total_findings,
                            current_activity=f"Extracted {len(findings)} findings from {page.domain}",
                        ))
                
                # ── 2d. Synthesize this step ──
                context_graph.update_node(step_node_id, {
                    "status": "completed",
                    "findings_count": step_findings_count,
                })
                
                if step_findings_count > 0 and not self.is_cancelled(session_id):
                    step_synthesis = await rlm.synthesize_step(
                        step_id=step_node_id,
                        step_title=step.title,
                        research_query=query,
                    )
                    step_syntheses.append({"title": step.title, "synthesis": step_synthesis})
                else:
                    step_syntheses.append({"title": step.title, "synthesis": f"Limited findings for: {step.title}"})
                
                await emit(ResearchProgressEvent(
                    event_type="step_completed",
                    step_index=step_idx,
                    total_steps=len(plan.steps),
                    sources_scraped=total_sources_scraped,
                    total_findings=total_findings,
                    current_activity=f"Completed step: {step.title} ({step_findings_count} findings)",
                ))
            
            # ── Phase 3: FINAL SYNTHESIS ──────────────────
            return await self._finalize(
                session_id, query, context_graph, rlm,
                step_syntheses, start_time,
                cancelled=self.is_cancelled(session_id),
            )
            
        except Exception as e:
            logger.error(f"[DeepResearch] Research failed: {e}")
            self._sessions[session_id]["status"] = ResearchStatus.ERROR
            
            await emit(ResearchProgressEvent(
                event_type="error",
                sources_scraped=self._sessions[session_id].get("sources_scraped", 0),
                total_findings=self._sessions[session_id].get("total_findings", 0),
                current_activity=f"Research error: {str(e)[:200]}",
            ))
            
            return ResearchResult(
                session_id=session_id,
                query=query,
                report=f"# Research Error\n\nAn error occurred during research: {str(e)}\n\nPartial findings may be available.",
                status=ResearchStatus.ERROR,
                research_time_ms=(time.time() - start_time) * 1000,
            )

    async def _finalize(
        self,
        session_id: str,
        query: str,
        context_graph: ResearchContextGraph,
        rlm: RecursiveResearchAgent,
        step_syntheses: List[dict],
        start_time: float,
        cancelled: bool = False,
    ) -> ResearchResult:
        """Perform final synthesis and return the research result."""
        
        self._sessions[session_id]["status"] = ResearchStatus.SYNTHESIZING
        
        # Emit synthesis starting event
        progress_cb = None
        if session_id in self._sessions:
            events = self._sessions[session_id]["progress_events"]
            # We'll add the event directly
            events.append(ResearchProgressEvent(
                event_type="synthesis_started",
                sources_scraped=context_graph.get_source_count(),
                total_findings=context_graph.get_findings_count(),
                current_activity="Synthesizing final research report..." + (" (early stop)" if cancelled else ""),
                thinking="Combining all findings into a comprehensive research report with source attribution.",
            ).model_dump())
        
        # Generate final report
        if step_syntheses:
            report = await rlm.synthesize_final(query, step_syntheses)
        else:
            report = f"# Research Report: {query}\n\nInsufficient data was gathered to generate a comprehensive report. Please try again with different search terms."
        
        # Gather sources
        sources = context_graph.get_all_sources()
        decision_trace = context_graph.get_decision_trace()
        graph_stats = context_graph.get_stats()
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        status = ResearchStatus.CANCELLED if cancelled else ResearchStatus.COMPLETE
        self._sessions[session_id]["status"] = status
        
        # Store to session
        self._sessions[session_id]["report"] = report
        self._sessions[session_id]["sources"] = [s.model_dump() for s in sources]
        self._sessions[session_id]["graph_data"] = context_graph.export_graph()
        
        # Emit completion
        events = self._sessions[session_id]["progress_events"]
        events.append(ResearchProgressEvent(
            event_type="research_complete",
            sources_scraped=graph_stats.get("sources", 0),
            total_findings=graph_stats.get("findings", 0),
            current_activity=f"Research {'stopped early' if cancelled else 'complete'}!",
            data={
                "research_time_ms": elapsed_ms,
                "graph_stats": graph_stats,
            },
        ).model_dump())
        
        # Clear cancellation flag
        self._cancelled.discard(session_id)
        
        result = ResearchResult(
            session_id=session_id,
            query=query,
            report=report,
            sources=sources,
            total_sources_scraped=graph_stats.get("sources", 0),
            total_findings=graph_stats.get("findings", 0),
            research_time_ms=elapsed_ms,
            status=status,
            decision_trace=decision_trace,
        )
        
        logger.info(
            f"[DeepResearch] {'Completed' if not cancelled else 'Cancelled'} session {session_id}: "
            f"{graph_stats.get('sources', 0)} sources, {graph_stats.get('findings', 0)} findings, "
            f"{elapsed_ms:.0f}ms"
        )
        
        return result
