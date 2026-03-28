"""
Recursive Language Model (RLM) Engine — Core implementation of the
Zhang et al. paradigm for infinite context processing.

The RLM treats large content as an external environment variable rather than
feeding it directly into the LLM context window.  It recursively decomposes
content into manageable chunks, extracts findings via sub-agent calls, and
synthesises using map-reduce patterns.

Architectural improvements over the original implementation:
- ALL LLM calls are non-blocking (async_complete via run_in_executor) so the
  FastAPI event loop is never stalled.  The research session runs as an
  asyncio.create_task(); any blocking call inside it prevents /status polling,
  SSE flushing, and other HTTP requests from being served — making the feature
  appear broken from the frontend's perspective.
- Chunks are extracted in PARALLEL using asyncio.gather, bounded by a semaphore
  to prevent rate-limit hits.
- Retry logic with exponential backoff handles transient LLM failures.
- Cross-chunk finding deduplication removes redundant content.
- Robust JSON parser uses bracket-depth matching to recover from LLM formatting
  quirks (extra text, nested objects, trailing commas, etc.).
"""

import asyncio
import json
import uuid
import re
from typing import List, Optional, Dict
from loguru import logger

from backend.core.deep_research.models import Finding
from backend.core.deep_research.context_graph import ResearchContextGraph


# Maximum concurrent LLM calls across all chunk-extraction coroutines.
# Keeps us below typical provider rate limits (30 RPM for free tiers).
_LLM_CONCURRENCY = 3


class RecursiveResearchAgent:
    """
    RLM-based agent for extracting and synthesising research findings.

    Implements the 5-phase RLM algorithm:
    1. Environment Initialisation — content is kept as external variables,
       not injected wholesale into the LLM context window.
    2. Programmatic Examination   — inspect content length / structure.
    3. Programmatic Decomposition — semantic chunking (paragraph-preferred).
    4. Recursive Invocation       — async LLM call per chunk (parallel).
    5. Synthesis                  — map-reduce aggregation + final report.

    Key design choices:
    ─ LLM calls use async_complete() so they run in a thread-pool and never
      block the event loop.
    ─ Chunk extraction is parallelised via asyncio.gather with a semaphore.
    ─ Findings are deduplicated across chunks before being stored in the graph.
    ─ Graceful degradation: partial results are returned on individual failures.
    """

    CHUNK_SIZE = 6000        # chars per chunk — large enough for rich context
    CHUNK_OVERLAP = 400      # overlap preserves cross-boundary context
    MAX_CHUNKS = 8           # cap to prevent token explosion on huge pages
    MAX_FINDINGS_PER_CHUNK = 10
    EXTRACTION_RETRIES = 2   # transient-failure retries per chunk

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
        # Shared semaphore across ALL concurrent chunk extractions in this session
        self._llm_semaphore = asyncio.Semaphore(_LLM_CONCURRENCY)

    # ═══════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════

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
        Entry point: extract structured findings from a scraped page.

        Uses RLM decomposition for pages larger than CHUNK_SIZE:
        ─ Small pages  → single async LLM extraction call
        ─ Large pages  → decompose into chunks → parallel async extraction → merge
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

        # Deduplicate before writing to the graph
        findings = self._deduplicate_findings(findings)

        for finding in findings:
            self.graph.add_finding(finding, source_id, step_node_id)

        return findings

    async def synthesize_step(
        self,
        step_id: str,
        step_title: str,
        research_query: str,
    ) -> str:
        """
        Map-reduce synthesis of all findings extracted during one research step.
        Groups findings by theme, preserves technical details, attributes sources.
        """
        findings = self.graph.get_findings_for_step(step_id)

        if not findings:
            return f"No findings were extracted for step: {step_title}"

        # Cap at top-30 by importance, then truncate total text to prevent 413 errors
        if len(findings) > 30:
            findings = sorted(
                findings, key=lambda f: f.get("importance", 0.5), reverse=True
            )[:30]

        findings_text = self._format_findings(findings)

        # Hard cap on findings_text size (Groq free-tier models have tight limits)
        if len(findings_text) > 12000:
            findings_text = findings_text[:12000] + "\n\n[Further findings truncated to fit context]"

        system_prompt = (
            "You are an expert research synthesiser. Given extracted findings from multiple "
            "sources, create a DEEP, DETAILED synthesis that:\n\n"
            "1. Organises findings by theme or sub-topic (NOT by source)\n"
            "2. Preserves ALL specific technical details — formulas, numbers, model names, "
            "benchmark scores\n"
            "3. Identifies patterns, contradictions, and areas of consensus\n"
            "4. Establishes connections between findings from different sources\n"
            "5. Maintains inline source attribution using [Source Title] format\n"
            "6. Provides expert-level analysis on the significance of findings\n\n"
            "Requirements:\n"
            "- Write 4-8 substantive paragraphs\n"
            "- Every sentence must convey specific information — no filler or generic statements\n"
            "- Include technical vocabulary and domain-specific terminology\n"
            "- Include formulas, equations, numbers, metrics if the findings contain them\n"
            "- DO NOT start with 'The findings show...' or similar generic openers\n"
            "- DO NOT hallucinate facts not present in the provided findings"
        )

        user_prompt = (
            f'Research question: "{research_query}"\n'
            f'Step: "{step_title}"\n\n'
            f"Findings from {len(findings)} extractions:\n"
            f"{findings_text}\n\n"
            "Produce a deep, comprehensive synthesis of these findings:"
        )

        try:
            synthesis = await self.llm.async_complete(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                max_tokens=3000,
                temperature=0.4,
            )

            finding_ids = [f.get("id", "") for f in findings if f.get("id")]
            self.graph.add_synthesis(content=synthesis, finding_ids=finding_ids, depth=0)
            return synthesis

        except Exception as e:
            logger.error(f"[RLM] Step synthesis failed for '{step_title}': {e}")
            return f"Synthesis failed for step: {step_title}. Error: {str(e)[:120]}"

    async def synthesize_final(
        self,
        research_query: str,
        step_syntheses: List[dict],
        report_structure: Optional[List[Dict]] = None,
    ) -> str:
        """
        Final recursive synthesis across all steps.
        Generates a comprehensive expert-level research report (3 000–8 000 words).
        """
        # Only include sources that actually yielded findings — prevents the LLM
        # from hallucinating content "about" search-result URLs that were never
        # scraped or were relevance-gated out (which was causing Japanese Q&A /
        # product pages to appear as citations in the output).
        sources = self.graph.get_all_sources(with_findings_only=True)
        graph_stats = self.graph.get_stats()
        structure = report_structure or self.report_structure

        # ── Truncate individual step syntheses to avoid 413 / context-overflow ──
        # Cap each step synthesis at 2 000 chars and the full block at 20 000 chars.
        MAX_CHARS_PER_STEP = 2000
        MAX_TOTAL_STEPS_CHARS = 20000
        steps_text = ""
        for s in step_syntheses:
            title = s.get("title", "Step")
            synthesis = s.get("synthesis", "No synthesis available.")
            if len(synthesis) > MAX_CHARS_PER_STEP:
                synthesis = synthesis[:MAX_CHARS_PER_STEP] + "... [truncated]"
            block = f"\n### {title}\n{synthesis}\n"
            if len(steps_text) + len(block) > MAX_TOTAL_STEPS_CHARS:
                steps_text += "\n[Additional step syntheses omitted to stay within context limits.]\n"
                break
            steps_text += block

        source_list = ""
        for i, src in enumerate(sources, 1):
            label = src.title or src.domain or src.url
            source_list += f"{i}. [{label}]({src.url}) — {src.findings_count} findings\n"

        structure_instructions = ""
        if structure:
            structure_instructions = (
                "\n\nUSE THE FOLLOWING CUSTOM REPORT STRUCTURE (these sections are MANDATORY):\n"
            )
            for i, section in enumerate(structure, 1):
                title = section.get("section_title", f"Section {i}")
                purpose = section.get("section_purpose", "")
                expected = section.get("expected_content", "")
                structure_instructions += f"\n## {i}. {title}\n"
                if purpose:
                    structure_instructions += f"   Purpose: {purpose}\n"
                if expected:
                    structure_instructions += f"   Expected content: {expected}\n"
            structure_instructions += (
                "\nYou MUST use these exact section titles. "
                "Do NOT add generic sections like 'Introduction', 'Background', "
                "'Conclusion', or 'Appendix'."
            )

        system_prompt = (
            "You are a world-class research analyst producing a comprehensive, expert-level "
            "research report.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. The report must be 3 000–8 000 words — genuinely comprehensive and deeply detailed\n"
            "2. Every claim MUST be grounded in the research findings provided — absolutely no "
            "hallucination\n"
            "3. Include specific technical details: formulas, numbers, architecture names, "
            "benchmark scores, author names\n"
            "4. Use inline citations in [Source Title] format\n"
            "5. Include comparison tables where appropriate (markdown format)\n"
            "6. Write in an authoritative, expert academic tone\n"
            "7. Each section should contain multiple paragraphs with substantial technical content\n"
            "8. Include code snippets or pseudocode if the topic involves algorithms\n"
            "9. Include mathematical formulations if the topic involves formal methods\n"
            "10. DO NOT use generic filler paragraphs — every paragraph must add specific value\n\n"
            "STRUCTURE RULES:\n"
            "- DO NOT use generic sections ('Introduction', 'Background and Context', "
            "'Limitations', 'Conclusion', 'Appendix')\n"
            "- Use topic-specific section titles that reflect the actual content\n"
            "- Each section should be 400–1 000 words\n"
            "- Use ## for main sections and ### for subsections\n"
            f"{structure_instructions}\n\n"
            "FORMATTING:\n"
            "- Use markdown: **bold** for key terms, `code` for technical names, "
            "tables for comparisons\n"
            "- Use numbered lists for steps/procedures, bullet points for features/properties\n"
            "- Include horizontal rules (---) between major sections for readability"
        )

        user_prompt = (
            f'Research question: "{research_query}"\n\n'
            f"Research covered {graph_stats.get('sources', 0)} sources with "
            f"{graph_stats.get('findings', 0)} total findings across "
            f"{len(step_syntheses)} research steps.\n\n"
            f"Per-step research syntheses:\n{steps_text}\n\n"
            f"Sources consulted:\n{source_list}\n\n"
            "Write the comprehensive research report. "
            "Remember: 3 000–8 000 words, deeply technical, every claim attributed to sources:"
        )

        try:
            report = await self.llm.async_complete(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                max_tokens=8000,
                temperature=0.5,
            )

            step_synth_ids = [
                n.get("id") for n in self.graph.get_synthesis_chain() if n.get("id")
            ]
            self.graph.add_synthesis(content=report, finding_ids=step_synth_ids, depth=1)
            return report

        except Exception as e:
            logger.error(f"[RLM] Final synthesis failed: {e}")
            fallback = f"# Research Report: {research_query}\n\n"
            for s in step_syntheses:
                fallback += f"## {s.get('title', 'Step')}\n\n{s.get('synthesis', '')}\n\n"
            return fallback

    # ═══════════════════════════════════════════════════
    # RLM Core: Recursive Extraction
    # ═══════════════════════════════════════════════════

    async def _recursive_extract(
        self,
        content: str,
        query: str,
        source_url: str,
        source_title: str,
        depth: int,
    ) -> List[Finding]:
        """
        Phase 4 of the RLM algorithm — the llm_query function.

        Base case:  content fits in one chunk, or we have reached max recursion depth
                    → single async LLM extraction call.
        Recursive:  decompose into chunks → launch all chunk extractions in parallel
                    via asyncio.gather → merge results.

        The semaphore in _extract_from_chunk ensures we never exceed
        _LLM_CONCURRENCY simultaneous provider calls.
        """
        # ── Base case ──
        if len(content) <= self.CHUNK_SIZE or depth >= self.depth_limit:
            return await self._extract_from_chunk(
                chunk=content[: self.CHUNK_SIZE * 2],  # allow slight overflow at max depth
                query=query,
                source_url=source_url,
                source_title=source_title,
                depth=depth,
            )

        # ── Recursive case: Phase 3 Decomposition + Phase 4 parallel map ──
        chunks = self._chunk_content(content)

        # Prioritise chunks when there are too many: first 40% + middle 40% + last 20%
        if len(chunks) > self.MAX_CHUNKS:
            n = self.MAX_CHUNKS
            first_n = max(1, n * 2 // 5)
            last_n = max(1, n // 5)
            mid_n = n - first_n - last_n
            mid_start = len(chunks) // 2 - mid_n // 2
            chunks = (
                chunks[:first_n]
                + chunks[mid_start: mid_start + mid_n]
                + chunks[-last_n:]
            )

        logger.debug(
            f"[RLM] Depth {depth}: parallel extraction over {len(chunks)} chunks "
            f"(source: {source_url[:60]})"
        )

        tasks = [
            self._recursive_extract(
                content=chunk,
                query=query,
                source_url=source_url,
                source_title=source_title,
                depth=depth + 1,
            )
            for chunk in chunks
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_findings: List[Finding] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"[RLM] Chunk {i} raised an exception: {result}")
            elif isinstance(result, list):
                all_findings.extend(result)

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
        Extract findings from a single text chunk via a single async LLM call.
        This is the RLM 'sub-agent'.

        Uses the shared semaphore to cap concurrent provider calls and implements
        exponential-backoff retry for transient errors (rate limits, timeouts).
        """
        system_prompt = (
            "You are an expert research analyst performing deep extraction from academic "
            "and technical text.\n\n"
            "For the given research question, extract 5-10 detailed findings from the "
            "provided text. Each finding should be:\n"
            "- A substantial, self-contained piece of information (2-4 sentences minimum)\n"
            "- Include specific technical details: numbers, formulas, architecture names, "
            "method names, author names\n"
            "- Provide enough context that the finding is useful standalone\n"
            "- Accurately classified by type\n\n"
            "Types:\n"
            '- "definition": Formal definitions, terminology explanations\n'
            '- "methodology": Algorithms, architectures, techniques, procedures\n'
            '- "result": Empirical results, benchmark scores, performance metrics, statistics\n'
            '- "comparison": Comparisons between approaches, trade-offs, advantages/disadvantages\n'
            '- "insight": Novel observations, implications, connections between ideas\n'
            '- "citation": References to other important works, author attributions\n'
            '- "technical_detail": Implementation specifics, hyperparameters, design choices\n\n'
            "Respond ONLY with a JSON array — no explanation, no markdown preamble. Format:\n"
            "[\n"
            "  {\n"
            '    "content": "Detailed finding with specific technical details.",\n'
            '    "extraction_type": "definition|methodology|result|comparison|insight|'
            'citation|technical_detail",\n'
            '    "importance": 0.0\n'
            "  }\n"
            "]\n\n"
            "RULES:\n"
            "- Extract MORE findings, not fewer — every substantive claim should be captured.\n"
            "- Include specific numbers, percentages, model names, dataset names.\n"
            "- If the text contains formulas or equations, include them in the finding.\n"
            "- If the text references specific papers or authors, include those attributions.\n"
            "- importance: 0.9 = critical finding, 0.7 = important, 0.5 = useful, "
            "0.3 = peripheral.\n"
            "- If the text contains NO information relevant to the research question, "
            "return exactly: []"
        )

        user_prompt = (
            f'Research question: "{query}"\n\n'
            f"Source: {source_title or 'Unknown'} ({source_url or 'Unknown URL'})\n\n"
            f"Text to analyse (recursion depth {depth}):\n"
            "---\n"
            f"{chunk}\n"
            "---\n\n"
            "Extract ALL key findings relevant to the research question. "
            "Return ONLY a JSON array:"
        )

        for attempt in range(self.EXTRACTION_RETRIES + 1):
            try:
                async with self._llm_semaphore:
                    response = await self.llm.async_complete(
                        messages=[{"role": "user", "content": user_prompt}],
                        system_prompt=system_prompt,
                        max_tokens=3000,
                        temperature=0.2,
                    )

                findings_data = self._parse_json_response(response)

                if findings_data is None:
                    if attempt < self.EXTRACTION_RETRIES:
                        logger.debug(
                            f"[RLM] JSON parse failed (attempt {attempt + 1}/{self.EXTRACTION_RETRIES + 1}), retrying"
                        )
                        await asyncio.sleep(0.5)
                        continue
                    logger.warning(
                        f"[RLM] Could not parse JSON from LLM response after all attempts. "
                        f"Response preview: {str(response)[:200]}"
                    )
                    return []

                if not isinstance(findings_data, list):
                    return []

                findings: List[Finding] = []
                for fd in findings_data[: self.MAX_FINDINGS_PER_CHUNK]:
                    if not isinstance(fd, dict):
                        continue
                    content_text = fd.get("content", "").strip()
                    if not content_text or len(content_text) < 20:
                        continue
                    findings.append(
                        Finding(
                            id=str(uuid.uuid4()),
                            content=content_text,
                            source_url=source_url,
                            source_title=source_title,
                            extraction_type=fd.get("extraction_type", "fact"),
                            importance=min(
                                1.0, max(0.0, float(fd.get("importance", 0.5)))
                            ),
                            depth=depth,
                        )
                    )

                logger.debug(
                    f"[RLM] Depth {depth}: extracted {len(findings)} findings "
                    f"from chunk ({len(chunk)} chars)"
                )
                return findings

            except Exception as e:
                error_msg = str(e)
                if attempt < self.EXTRACTION_RETRIES:
                    wait = 1.5 ** attempt  # 1 s, 1.5 s
                    logger.warning(
                        f"[RLM] Extraction attempt {attempt + 1} failed: "
                        f"{error_msg[:120]}. Retrying in {wait:.1f}s…"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"[RLM] Extraction failed after {self.EXTRACTION_RETRIES + 1} "
                        f"attempts for chunk from {source_url[:60]}: {error_msg[:200]}"
                    )
                    return []

        return []

    # ═══════════════════════════════════════════════════
    # Helper methods
    # ═══════════════════════════════════════════════════

    def _chunk_content(self, content: str) -> List[str]:
        """
        Phase 3: Programmatic Decomposition.

        Splits content into semantically meaningful chunks, preferring paragraph
        boundaries, falling back to sentence splits, then hard character splits.
        """
        if len(content) <= self.CHUNK_SIZE:
            return [content]

        chunks: List[str] = []
        paragraphs = re.split(r"\n\s*\n", content)
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.CHUNK_SIZE:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                if len(para) > self.CHUNK_SIZE:
                    # Sentence-level split for very long paragraphs
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= self.CHUNK_SIZE:
                            current_chunk += sent + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            if len(sent) > self.CHUNK_SIZE:
                                # Hard character split with overlap
                                for i in range(
                                    0,
                                    len(sent),
                                    self.CHUNK_SIZE - self.CHUNK_OVERLAP,
                                ):
                                    chunks.append(sent[i : i + self.CHUNK_SIZE])
                                current_chunk = ""
                            else:
                                current_chunk = sent + " "
                else:
                    current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [content[: self.CHUNK_SIZE]]

    def _deduplicate_findings(self, findings: List[Finding]) -> List[Finding]:
        """
        Remove near-duplicate findings extracted from overlapping chunks.
        Uses Jaccard similarity on word sets (first 120 chars as proxy).
        """
        if len(findings) <= 1:
            return findings

        unique: List[Finding] = []
        seen_word_sets: List[frozenset] = []

        for f in findings:
            words = frozenset(re.findall(r"[a-z]{3,}", f.content[:120].lower()))
            is_dup = False
            for seen in seen_word_sets:
                union = seen | words
                if union:
                    jaccard = len(seen & words) / len(union)
                    if jaccard > 0.65:
                        is_dup = True
                        break
            if not is_dup:
                unique.append(f)
                seen_word_sets.append(words)

        removed = len(findings) - len(unique)
        if removed:
            logger.debug(f"[RLM] Deduplication: removed {removed} near-duplicate findings")

        return unique

    def _format_findings(self, findings: List[dict]) -> str:
        """Format findings list as annotated text for LLM synthesis context."""
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
        """
        Robust JSON parser with five recovery strategies:
        1. Strip markdown code fences, then direct parse.
        2. Bracket-depth matching for arrays (handles leading/trailing text).
        3. Bracket-depth matching for objects.
        4. Remove trailing commas before ] or } (common LLM mistake).
        5. Fallback: None (caller handles gracefully).
        """
        if not response:
            return None

        text = response.strip()

        # Strategy 1: strip code fences then direct parse
        cleaned = re.sub(r"```(?:json)?\s*\n?", "", text)
        cleaned = re.sub(r"\n?```", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 2: bracket-depth array extraction
        array_result = self._extract_by_depth(cleaned, "[", "]")
        if array_result is not None:
            return array_result

        # Strategy 3: bracket-depth object extraction
        obj_result = self._extract_by_depth(cleaned, "{", "}")
        if obj_result is not None:
            return obj_result

        # Strategy 4: fix trailing commas then retry
        fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        array_result2 = self._extract_by_depth(fixed, "[", "]")
        if array_result2 is not None:
            return array_result2

        return None

    @staticmethod
    def _extract_by_depth(text: str, open_c: str, close_c: str):
        """
        Find the first balanced bracket pair in *text* and parse the JSON inside.
        Correctly handles nested structures unlike a naive rfind approach.
        """
        start = text.find(open_c)
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None

        return None
