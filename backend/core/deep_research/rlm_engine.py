"""
Recursive Language Model (RLM) Engine — Core implementation of the 
Zhang et al. paradigm for infinite context processing.

The RLM treats large content as an external environment variable rather than
feeding it directly into the LLM context window. It recursively decomposes
content into manageable chunks, extracts findings via sub-agent calls,
and synthesizes using map-reduce patterns.
"""

import uuid
import re
from typing import List, Optional
from loguru import logger

from backend.core.deep_research.models import Finding
from backend.core.deep_research.context_graph import ResearchContextGraph


class RecursiveResearchAgent:
    """
    RLM-based agent for extracting and synthesizing research findings.
    
    Implements the 5-phase RLM algorithm:
    1. Environment Initialization — content loaded as external variables
    2. Programmatic Examination — peek at content structure
    3. Programmatic Decomposition — semantic chunking
    4. Recursive Invocation — llm_query for each chunk
    5. Synthesis — map-reduce aggregation of findings
    
    Emergent behaviors:
    - Map-Reduce: summarize chunks → merge summaries
    - Semantic Binary Search: for targeted fact retrieval
    """

    # Maximum chunk size that fits comfortably in a single LLM call
    CHUNK_SIZE = 4000  # characters
    CHUNK_OVERLAP = 200  # overlap for context continuity

    def __init__(
        self,
        llm_router,
        context_graph: ResearchContextGraph,
        depth_limit: int = 3
    ):
        self.llm = llm_router
        self.graph = context_graph
        self.depth_limit = depth_limit

    async def extract_findings(
        self,
        page_content: str,
        research_query: str,
        source_id: str,
        source_url: str = "",
        source_title: str = "",
        step_node_id: Optional[str] = None,
    ) -> List[Finding]:
        """
        Extract structured findings from a scraped page using RLM decomposition.
        
        This is the core RLM function:
        - If content fits in one chunk → direct extraction
        - If content is too large → decompose into chunks, extract per chunk, merge
        
        Args:
            page_content: Full text content of the page
            research_query: The research question guiding extraction
            source_id: Graph node ID of the web source
            source_url: URL for attribution
            source_title: Title for attribution
            step_node_id: Optional step node for graph linking
            
        Returns:
            List of extracted findings
        """
        if not page_content or len(page_content.strip()) < 50:
            return []

        findings = await self._recursive_extract(
            content=page_content,
            query=research_query,
            source_url=source_url,
            source_title=source_title,
            depth=0,
        )

        # Add findings to the context graph
        for finding in findings:
            self.graph.add_finding(finding, source_id, step_node_id)

        return findings

    async def _recursive_extract(
        self,
        content: str,
        query: str,
        source_url: str,
        source_title: str,
        depth: int,
    ) -> List[Finding]:
        """
        Recursive extraction — the llm_query function from the RLM paper.
        
        Phase 4 of the RLM algorithm:
        - If content fits → direct LLM extraction (base case)
        - If too large → decompose → map extraction over chunks → reduce
        """
        # Base case: content fits in one chunk
        if len(content) <= self.CHUNK_SIZE or depth >= self.depth_limit:
            return await self._extract_from_chunk(
                content[:self.CHUNK_SIZE * 2],  # Allow some overflow at max depth
                query,
                source_url,
                source_title,
                depth,
            )

        # Recursive case: decompose and map-reduce
        # Phase 3: Programmatic Decomposition
        chunks = self._chunk_content(content)
        
        logger.debug(f"[RLM] Depth {depth}: Decomposed into {len(chunks)} chunks")

        # Phase 4: Map — extract findings from each chunk
        all_findings = []
        for chunk in chunks:
            chunk_findings = await self._recursive_extract(
                content=chunk,
                query=query,
                source_url=source_url,
                source_title=source_title,
                depth=depth + 1,
            )
            all_findings.extend(chunk_findings)

        return all_findings

    async def _extract_from_chunk(
        self,
        chunk: str,
        query: str,
        source_url: str,
        source_title: str,
        depth: int,
    ) -> List[Finding]:
        """
        Extract findings from a single chunk via LLM call.
        This is the "sub-agent" in the RLM paradigm.
        """
        system_prompt = """You are a research analyst extracting key findings from text.

For the given research question, extract 2-5 key findings from the provided text.
Each finding should be a self-contained, factual statement or insight.

Respond ONLY with a JSON array in this format:
[
  {
    "content": "Clear, concise finding statement",
    "extraction_type": "fact|statistic|quote|analysis|definition",
    "importance": 0.0-1.0
  }
]

If the text contains no relevant information, return an empty array: []"""

        user_prompt = f"""Research question: "{query}"

Source: {source_title} ({source_url})

Text to analyze:
---
{chunk}
---

Extract the key findings relevant to the research question:"""

        try:
            response = self.llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                max_tokens=1500,
                temperature=0.3,
            )
            
            findings_data = self._parse_json_response(response)
            
            if not isinstance(findings_data, list):
                return []
            
            findings = []
            for fd in findings_data[:5]:  # Cap at 5 per chunk
                if isinstance(fd, dict) and fd.get("content"):
                    findings.append(Finding(
                        id=str(uuid.uuid4()),
                        content=fd["content"],
                        source_url=source_url,
                        source_title=source_title,
                        extraction_type=fd.get("extraction_type", "fact"),
                        importance=min(1.0, max(0.0, float(fd.get("importance", 0.5)))),
                        depth=depth,
                    ))
            
            return findings
            
        except Exception as e:
            logger.warning(f"[RLM] Extraction failed at depth {depth}: {e}")
            return []

    async def synthesize_step(
        self,
        step_id: str,
        step_title: str,
        research_query: str,
    ) -> str:
        """
        Map-reduce synthesis of all findings for one research step.
        
        Phase 5 of the RLM algorithm:
        - Aggregates findings from all sources in this step
        - Produces a per-step synthesis
        """
        findings = self.graph.get_findings_for_step(step_id)
        
        if not findings:
            return f"No findings were extracted for step: {step_title}"

        # Format findings as context
        findings_text = self._format_findings(findings)

        system_prompt = """You are a research synthesizer. Given a set of findings from multiple sources, create a concise synthesis that:
1. Identifies the key themes and insights
2. Notes any contradictions or debates
3. Highlights the most important facts
4. Maintains source attribution

Write in clear, professional prose (2-4 paragraphs). Do not use generic filler — every sentence should convey specific information."""

        user_prompt = f"""Research question: "{research_query}"
Step: "{step_title}"

Findings from {len(findings)} extractions:
{findings_text}

Synthesize these findings:"""

        try:
            synthesis = self.llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                max_tokens=1500,
                temperature=0.4,
            )
            
            # Add synthesis to graph
            finding_ids = [f.get("id", "") for f in findings if f.get("id")]
            self.graph.add_synthesis(
                content=synthesis,
                finding_ids=finding_ids,
                depth=0,
            )
            
            return synthesis
            
        except Exception as e:
            logger.error(f"[RLM] Step synthesis failed: {e}")
            return f"Synthesis failed for step: {step_title}"

    async def synthesize_final(
        self,
        research_query: str,
        step_syntheses: List[dict],
    ) -> str:
        """
        Final recursive synthesis across all steps.
        
        This produces the comprehensive research report by combining
        all per-step syntheses with the full context graph.
        """
        sources = self.graph.get_all_sources()
        graph_stats = self.graph.get_stats()

        # Build the combined synthesis input
        steps_text = ""
        for s in step_syntheses:
            steps_text += f"\n### {s.get('title', 'Step')}\n{s.get('synthesis', 'No synthesis available.')}\n"

        # Format source list
        source_list = ""
        for i, src in enumerate(sources, 1):
            source_list += f"{i}. [{src.title}]({src.url}) — {src.findings_count} findings\n"

        system_prompt = """You are a senior research analyst producing a comprehensive research report.

Requirements:
1. Write a well-structured report with clear sections and headers using markdown
2. Start with an executive summary without the label "Executive Summary"
3. Organize the body by themes/topics, NOT by research steps
4. Include specific facts, statistics, and key insights from the research. Inculcate tabular format if need be
5. Note areas of consensus and disagreement among sources
6. Conclude with key takeaways and implications
7. Maintain an academic/professional tone throughout
8. Use markdown formatting: headers (##, ###), bold, bullet points, and numbered lists
9. Reference sources where appropriate using [Source Title](URL) format
10. EVERY claim should be grounded in the research findings — no hallucination

The report should be thorough (3000-6000 words) and provide genuinely useful insights."""

        user_prompt = f"""Research question: "{research_query}"

Research covered {graph_stats.get('sources', 0)} sources with {graph_stats.get('findings', 0)} total findings.

Per-step research syntheses:
{steps_text}


Write the comprehensive research report:"""

        try:
            report = self.llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                max_tokens=4000,
                temperature=0.5,
            )
            
            # Add final synthesis to graph
            step_synth_ids = [
                sid for sid in 
                [n.get("id") for n in self.graph.get_synthesis_chain()]
            ]
            self.graph.add_synthesis(
                content=report,
                finding_ids=step_synth_ids,
                depth=1,  # Final synthesis is depth 1
            )
            
            # Append sources section if not already included
            if "## Sources" not in report and "## References" not in report:
                report += f"\n\n---\n\n## Sources\n\n{source_list}"
            
            return report
            
        except Exception as e:
            logger.error(f"[RLM] Final synthesis failed: {e}")
            # Fallback: concatenate step syntheses
            fallback = f"# Research Report: {research_query}\n\n"
            for s in step_syntheses:
                fallback += f"## {s.get('title', 'Step')}\n\n{s.get('synthesis', '')}\n\n"
            fallback += f"\n---\n\n## Sources\n\n{source_list}"
            return fallback

    # ═══════════════════════════════════════════════════
    # Helper methods
    # ═══════════════════════════════════════════════════

    def _chunk_content(self, content: str) -> List[str]:
        """
        Phase 3: Programmatic Decomposition.
        
        Splits content into semantically meaningful chunks.
        Prefers splitting on paragraph boundaries, falls back to sentence/character splits.
        """
        if len(content) <= self.CHUNK_SIZE:
            return [content]

        chunks = []
        
        # Try splitting on double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.CHUNK_SIZE:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If single paragraph is too large, split it further
                if len(para) > self.CHUNK_SIZE:
                    # Split on sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= self.CHUNK_SIZE:
                            current_chunk += sent + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            # If single sentence is still too large, hard-split
                            if len(sent) > self.CHUNK_SIZE:
                                for i in range(0, len(sent), self.CHUNK_SIZE - self.CHUNK_OVERLAP):
                                    chunks.append(sent[i:i + self.CHUNK_SIZE])
                                current_chunk = ""
                            else:
                                current_chunk = sent + " "
                else:
                    current_chunk = para + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [content[:self.CHUNK_SIZE]]

    def _format_findings(self, findings: List[dict]) -> str:
        """Format findings list as text for LLM context."""
        parts = []
        for i, f in enumerate(findings, 1):
            importance = f.get("importance", 0.5)
            source = f.get("source_title") or f.get("source_url", "Unknown")
            parts.append(
                f"[{i}] (importance: {importance:.1f}, source: {source})\n"
                f"    {f.get('content', '')}"
            )
        return "\n\n".join(parts)

    def _parse_json_response(self, response: str):
        """Parse JSON from LLM response."""
        if not response:
            return None
        
        text = response.strip()
        
        # Remove markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        
        try:
            return __import__("json").loads(text)
        except Exception:
            pass
        
        # Find JSON structures
        for start_c, end_c in [('{', '}'), ('[', ']')]:
            start = text.find(start_c)
            end = text.rfind(end_c)
            if start != -1 and end > start:
                try:
                    return __import__("json").loads(text[start:end + 1])
                except Exception:
                    continue
        
        return None
