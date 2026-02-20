"""
Recursive Language Model (RLM) Engine — Core implementation of the
Zhang et al. paradigm for infinite context processing.

The RLM treats large content as an external environment variable rather than
feeding it directly into the LLM context window. It recursively decomposes
content into manageable chunks, extracts findings via sub-agent calls,
and synthesizes using map-reduce patterns.

Enhanced with:
- Much deeper extraction prompts (8-15 findings per chunk instead of 2-5)
- Query-specific report format generation
- Multi-pass synthesis with recursive depth
- Source-attributed inline citations
- Expert-level academic writing quality
"""

import json
import uuid
import re
from typing import List, Optional, Dict
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
    - Multi-Resolution Extraction: variable detail per importance level
    """

    # Maximum chunk size that fits comfortably in a single LLM call
    CHUNK_SIZE = 5000  # characters — larger chunks for richer context
    CHUNK_OVERLAP = 300  # overlap for context continuity

    def __init__(
        self,
        llm_router,
        context_graph: ResearchContextGraph,
        depth_limit: int = 3,
        report_structure: Optional[List[Dict]] = None,
    ):
        self.llm = llm_router
        self.graph = context_graph
        self.depth_limit = depth_limit
        self.report_structure = report_structure or []

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

        Enhanced to extract 5-10 highly detailed findings per chunk.
        """
        system_prompt = """You are an expert research analyst performing deep extraction from academic and technical text.

For the given research question, extract 5-10 detailed findings from the provided text. Each finding should be:
- A substantial, self-contained piece of information (2-4 sentences minimum)
- Include specific technical details: numbers, formulas, architecture names, method names, author names
- Provide enough context that the finding is useful standalone
- Classify the type accurately

Types:
- "definition": Formal definitions, terminology explanations
- "methodology": Algorithms, architectures, techniques, procedures
- "result": Empirical results, benchmark scores, performance metrics, statistics
- "comparison": Comparisons between approaches, trade-offs, advantages/disadvantages
- "insight": Novel observations, implications, connections between ideas
- "citation": References to other important works, author attributions
- "technical_detail": Implementation specifics, hyperparameters, design choices

Respond ONLY with a JSON array in this format:
[
  {
    "content": "Detailed, substantial finding with specific technical details. Include numbers, names, and context.",
    "extraction_type": "definition|methodology|result|comparison|insight|citation|technical_detail",
    "importance": 0.0-1.0
  }
]

IMPORTANT:
- Extract MORE findings, not fewer. Every substantive claim should be captured.
- Include specific numbers, percentages, model names, dataset names, etc.
- If the text contains formulas or equations, include them in the finding.
- If the text references specific papers or authors, include those attributions.
- If the text contains NO relevant information, return an empty array: []"""

        user_prompt = f"""Research question: "{query}"

Source: {source_title} ({source_url})

Text to analyze:
---
{chunk}
---

Extract ALL key findings relevant to the research question. Be thorough and detailed:"""

        try:
            response = self.llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                max_tokens=3000,
                temperature=0.3,
            )

            findings_data = self._parse_json_response(response)

            if not isinstance(findings_data, list):
                return []

            findings = []
            for fd in findings_data[:10]:  # Cap at 10 per chunk
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
        Enhanced for much deeper, more detailed per-step synthesis.
        """
        findings = self.graph.get_findings_for_step(step_id)

        if not findings:
            return f"No findings were extracted for step: {step_title}"

        # Format findings as context
        findings_text = self._format_findings(findings)

        system_prompt = """You are an expert research synthesizer. Given a set of extracted findings from multiple sources, create a DEEP, DETAILED synthesis that:

1. Organizes findings by theme or sub-topic (NOT by source)
2. Preserves ALL specific technical details — formulas, numbers, model names, benchmark scores
3. Identifies patterns, contradictions, and areas of consensus
4. Establishes connections between findings from different sources
5. Maintains inline source attribution using [Source Title] format
6. Provides expert-level analysis and commentary on the significance of findings
7. Highlights methodological details, architectural innovations, and empirical evidence

Requirements:
- Write 4-8 substantive paragraphs (do NOT write a single short paragraph)
- Every sentence must convey specific information — no filler or generic statements
- Include technical vocabulary and domain-specific terminology
- If findings contain formulas or equations, include them
- If findings contain specific numbers or metrics, include them
- DO NOT start with "The findings show..." or similar generic openers"""

        user_prompt = f"""Research question: "{research_query}"
Step: "{step_title}"

Findings from {len(findings)} extractions:
{findings_text}

Produce a deep, comprehensive synthesis of these findings:"""

        try:
            synthesis = self.llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                max_tokens=3000,
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
        report_structure: Optional[List[Dict]] = None,
    ) -> str:
        """
        Final recursive synthesis across all steps.
        Uses the query-specific report structure if available.
        """
        sources = self.graph.get_all_sources()
        graph_stats = self.graph.get_stats()

        # Use provided structure or fall back to instance variable
        structure = report_structure or self.report_structure

        # Build the combined synthesis input
        steps_text = ""
        for s in step_syntheses:
            steps_text += f"\n### {s.get('title', 'Step')}\n{s.get('synthesis', 'No synthesis available.')}\n"

        # Format source list
        source_list = ""
        for i, src in enumerate(sources, 1):
            source_list += f"{i}. [{src.title}]({src.url}) — {src.findings_count} findings\n"

        # Build structure instructions
        structure_instructions = ""
        if structure:
            structure_instructions = "\n\nUSE THE FOLLOWING CUSTOM REPORT STRUCTURE (these sections are MANDATORY):\n"
            for i, section in enumerate(structure, 1):
                title = section.get("section_title", f"Section {i}")
                purpose = section.get("section_purpose", "")
                expected = section.get("expected_content", "")
                structure_instructions += f"\n## {i}. {title}\n"
                if purpose:
                    structure_instructions += f"   Purpose: {purpose}\n"
                if expected:
                    structure_instructions += f"   Expected content: {expected}\n"
            structure_instructions += "\nYou MUST use these exact section titles. Do NOT add generic sections like 'Introduction', 'Background', 'Conclusion', or 'Appendix'."

        system_prompt = f"""You are a world-class research analyst producing a comprehensive, expert-level research report.

CRITICAL REQUIREMENTS:
1. The report must be 3000-8000 words — genuinely comprehensive and deeply detailed
2. Every claim MUST be grounded in the research findings provided — absolutely no hallucination
3. Include specific technical details throughout: formulas, numbers, architecture names, benchmark scores, author names
4. Use inline citations in [Source Title](URL) format
5. Include comparison tables where appropriate (markdown format)
6. Write in an authoritative, expert academic tone
7. Each section should contain multiple paragraphs with substantial technical content
8. Include code snippets or pseudocode if the topic involves algorithms
9. Include mathematical formulations if the topic involves formal methods
10. DO NOT use generic filler paragraphs — every paragraph must add specific value

STRUCTURE RULES:
- DO NOT use generic sections ("Introduction", "Background and Context", "Limitations", "Conclusion", "Appendix")
- Instead, use topic-specific section titles that reflect the actual content
- Each section should be 400-1000 words
- Use headers (##) for main sections and sub-headers (###) for subsections
{structure_instructions}

FORMATTING:
- Use markdown: **bold** for key terms, `code` for technical names, tables for comparisons
- Use numbered lists for steps/procedures, bullet points for features/properties
- Include horizontal rules (---) between major sections for readability"""

        user_prompt = f"""Research question: "{research_query}"

Research covered {graph_stats.get('sources', 0)} sources with {graph_stats.get('findings', 0)} total findings across {len(step_syntheses)} research steps.

Per-step research syntheses:
{steps_text}



Write the comprehensive research report. Remember: 3000-8000 words, deeply technical, every claim attributed to sources:"""

        try:
            report = self.llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                max_tokens=8000,
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
            return report

        except Exception as e:
            logger.error(f"[RLM] Final synthesis failed: {e}")
            # Fallback: concatenate step syntheses
            fallback = f"# Research Report: {research_query}\n\n"
            for s in step_syntheses:
                fallback += f"## {s.get('title', 'Step')}\n\n{s.get('synthesis', '')}\n\n"
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
            ftype = f.get("extraction_type", "fact")
            parts.append(
                f"[{i}] (type: {ftype}, importance: {importance:.1f}, source: {source})\n"
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
            return json.loads(text)
        except Exception:
            pass

        # Find JSON structures
        for start_c, end_c in [('{', '}'), ('[', ']')]:
            start = text.find(start_c)
            end = text.rfind(end_c)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except Exception:
                    continue

        return None
