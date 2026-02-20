"""
Agentic Research Planner — Uses LLM to generate, adapt, and evaluate research plans.
Implements the PLAN phase of the deep research pipeline.
Now generates BOTH a research plan AND a query-specific report format/structure.
"""

import json
from typing import List, Optional, Dict
from loguru import logger

from backend.core.deep_research.models import (
    ResearchPlan, ResearchStepDef, ResearchConfig
)


class ResearchPlanner:
    """
    Generates and adapts multi-step research plans using the LLM.

    The planner:
    1. Analyzes the user's research query
    2. Generates an ordered plan of 8-12 research steps
    3. Creates targeted search queries for each step
    4. Generates a CUSTOM report format/structure tailored to the query
    5. Can adapt the plan mid-research based on findings
    6. Evaluates whether enough information has been gathered
    """

    def __init__(self, llm_router, config: Optional[ResearchConfig] = None):
        self.llm = llm_router
        self.config = config or ResearchConfig()

    async def generate_plan(self, query: str) -> ResearchPlan:
        """
        Generate a research plan with ordered steps, search queries, and a
        custom report structure tailored to the query.
        """
        max_steps = self.config.max_steps

        system_prompt = f"""You are a world-class research planning expert. Given a research query, you must create:
1. A comprehensive, deeply technical research plan with 8-12 steps.
2. A CUSTOM REPORT STRUCTURE tailored specifically to the query (NOT a generic format).

CRITICAL RULES FOR SEARCH QUERIES:
- Each step MUST have 4-6 specific, diverse search queries
- Use PLAIN NATURAL LANGUAGE only — absolutely NO site: operators, NO URL prefixes like "site:arxiv.org" or "site:medium.com"
- These site: operators do not work with the search backend and will return zero results
- Instead, include the source type as keywords: use "arxiv" or "research paper" or "survey paper" naturally in the query
- Example GOOD queries: "object detection transformer survey arxiv 2024", "DETR architecture paper", "YOLOv9 benchmark COCO results"
- Example BAD queries: "site:arxiv.org object detection", "site:medium.com DETR tutorial" — NEVER do this
- Include year ranges (2022, 2023, 2024, 2025) in queries to find recent work
- Mix query styles: broad surveys, specific model names, benchmark comparisons, tutorial/explanation queries
- At least one query per step should name a specific model, paper title, or author when known
- At least one query per step should target benchmark/evaluation results
- Vary query length: some 3-word queries, some 6-8 word queries

CRITICAL RULES FOR REPORT STRUCTURE:
- The report structure must be SPECIFIC to the query topic, not a generic template
- Do NOT use generic sections like "Introduction", "Background", "Limitations", "Conclusion", "Appendix"
- Instead, derive section titles from the actual subject matter
- Each section should have a clear purpose description and expected content
- Think of it as an expert writing a deep-dive technical article on this specific topic

For example, if the query is "What are recursive language models?", the structure might be:
- "Formal Recursive Definitions in Syntax & Semantics" (covering Chomsky hierarchy, recursive grammars)
- "From Recurrence to Recursion in Neural Architectures" (RvNNs, Tree-LSTMs, R-Transformers)
- "Modern Recursive Paradigms: Self-Referential LLM Agents" (Auto-GPT, recursive prompting chains)
- "Mathematical Frameworks for Recursive Computation" (fixed-point semantics, mu-recursion)
- "Benchmark Evaluations and Empirical Results" (performance on specific tasks)
- "Open Problems and Future Directions" (unsolved challenges, active research fronts)

Respond ONLY with valid JSON, no markdown or extra text.

Output format:
{{
  "strategy": "depth-first" or "breadth-first",
  "estimated_sources": <number>,
  "report_structure": [
    {{
      "section_title": "Query-specific section title",
      "section_purpose": "What this section should cover and why",
      "expected_content": "Types of content expected: definitions, comparisons, formulas, case studies, etc."
    }}
  ],
  "steps": [
    {{
      "step_index": 0,
      "title": "Step title (specific and technical)",
      "description": "What this step investigates in depth",
      "search_queries": ["plain query with arxiv keyword", "specific model name paper", "benchmark comparison query", "tutorial explanation query", "survey recent advances query"]
    }}
  ]
}}"""

        user_prompt = f"""Create a highly detailed research plan AND custom report structure for this query:

\"{query}\"

Requirements:
- The plan must be thorough enough to produce a 5000+ word expert report
- Search queries MUST be plain natural language — no site: operators whatsoever
- Include "arxiv", "paper", "survey", "benchmark", "tutorial" as natural keywords in queries
- The report structure must be specific to the subject matter, NOT generic
- Each step should explore a distinct facet of the topic with surgical precision"""

        try:
            response = self.llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                max_tokens=3000,
                temperature=0.4,
            )

            plan_data = self._parse_json_response(response)

            if not plan_data or "steps" not in plan_data:
                logger.warning("[ResearchPlanner] Failed to parse plan, using fallback")
                return self._fallback_plan(query)

            steps = []
            for i, step_data in enumerate(plan_data["steps"][:max_steps]):
                # Sanitize queries: strip any site: operators that slipped through
                raw_queries = step_data.get("search_queries", [query])
                clean_queries = [self._sanitize_query(q) for q in raw_queries]
                steps.append(ResearchStepDef(
                    step_index=i,
                    title=step_data.get("title", f"Research Step {i+1}"),
                    description=step_data.get("description", ""),
                    search_queries=clean_queries,
                    status="pending",
                ))

            # Extract report structure
            report_structure = plan_data.get("report_structure", [])

            plan = ResearchPlan(
                steps=steps,
                strategy=plan_data.get("strategy", "depth-first"),
                estimated_sources=plan_data.get("estimated_sources", len(steps) * 5),
                report_structure=report_structure,
            )

            logger.info(f"[ResearchPlanner] Generated plan with {len(steps)} steps and {len(report_structure)} report sections")
            return plan

        except Exception as e:
            logger.error(f"[ResearchPlanner] Plan generation failed: {e}")
            return self._fallback_plan(query)

    async def generate_follow_up_queries(
        self,
        step_title: str,
        findings_summary: str,
        original_query: str
    ) -> List[str]:
        """
        Generate additional search queries based on findings so far.
        Uses plain natural language queries — no site: operators.
        """
        system_prompt = """You are a research assistant. Based on the current research findings, generate 3-5 follow-up search queries that:
1. Fill gaps in the current understanding with SPECIFIC technical details
2. Use PLAIN NATURAL LANGUAGE — absolutely no site: operators like "site:arxiv.org" or "site:medium.com"
3. Include source-type keywords naturally in the query text: "arxiv paper", "research survey", "technical tutorial", "benchmark evaluation"
4. Look for contradicting or supporting evidence from peer-reviewed sources
5. Search for benchmark comparisons, ablation studies, or empirical evaluations
6. Reference specific model names, authors, or paper titles when relevant

Respond ONLY with a JSON array of query strings, e.g.: ["query 1", "query 2", "query 3"]"""

        user_prompt = f"""Original research question: "{original_query}"
Current step: "{step_title}"

Findings so far:
{findings_summary[:3000]}

Generate targeted follow-up search queries (plain natural language only, no site: operators):"""

        try:
            response = self.llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.5,
            )

            queries = self._parse_json_response(response)
            if isinstance(queries, list):
                # Sanitize any operator queries that slipped through
                return [self._sanitize_query(str(q)) for q in queries[:5]]
            return []

        except Exception as e:
            logger.error(f"[ResearchPlanner] Follow-up query generation failed: {e}")
            return []

    async def evaluate_sufficiency(
        self,
        findings_summary: str,
        query: str,
        sources_count: int
    ) -> bool:
        """
        Evaluate whether enough information has been gathered to answer the query.
        Now requires higher standards: technical depth, source diversity, empirical evidence.
        """
        system_prompt = """You are a research evaluator. Determine if enough information has been gathered to write a COMPREHENSIVE, EXPERT-LEVEL research report (3000+ words).

You should answer "yes" ONLY if ALL of the following are true:
- Coverage: All major aspects of the topic are covered with technical depth
- Sources: Information comes from at least 10 diverse, high-quality sources (papers, articles, documentation)
- Depth: There are specific details — formulas, architectures, benchmark numbers, citations
- Balance: Multiple perspectives or approaches are represented
- Empirical Evidence: At least some empirical data, benchmarks, or case studies are included

If ANY of these criteria are not met, answer "no".

Respond with ONLY "yes" or "no"."""

        user_prompt = f"""Research question: "{query}"
Sources consulted: {sources_count}

Summary of findings:
{findings_summary[:4000]}

Is the research sufficient for a 3000+ word expert report? (yes/no)"""

        try:
            response = self.llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                max_tokens=10,
                temperature=0.1,
            )
            return response.strip().lower().startswith("yes")

        except Exception:
            return False  # Default to needing more research

    def _sanitize_query(self, query: str) -> str:
        """
        Remove site: operators and other unsupported search operators from queries.
        Converts operator-based queries to plain natural language equivalents.
        
        Examples:
            "site:arxiv.org object detection survey" -> "object detection survey arxiv"
            "site:medium.com DETR tutorial" -> "DETR tutorial"
            "site:neurips.cc attention mechanism" -> "attention mechanism neurips paper"
        """
        import re

        original = query.strip()

        # Extract and replace site: operators with natural keyword equivalents
        site_map = {
            "arxiv.org": "arxiv",
            "medium.com": "",
            "towardsdatascience.com": "",
            "neurips.cc": "neurips paper",
            "icml.cc": "icml paper",
            "aclanthology.org": "acl paper",
            "cvpr.thecvf.com": "cvpr paper",
            "ieeexplore.ieee.org": "ieee paper",
            "acm.org": "acm paper",
            "researchgate.net": "",
            "cv-foundation.org": "cvpr paper",
            "openreview.net": "openreview paper",
            "semanticscholar.org": "",
            "springer.com": "",
        }

        result = original

        # Match site:domain.tld patterns
        site_pattern = re.compile(r'site:([^\s]+)', re.IGNORECASE)
        matches = site_pattern.findall(result)

        for domain in matches:
            replacement = ""
            for known_domain, keyword in site_map.items():
                if known_domain in domain.lower():
                    replacement = keyword
                    break
            # Remove the site: operator
            result = re.sub(rf'site:{re.escape(domain)}\s*', '', result, flags=re.IGNORECASE)
            # Append the replacement keyword if it adds value
            if replacement and replacement not in result:
                result = result.strip() + f" {replacement}"

        # Clean up extra whitespace
        result = re.sub(r'\s+', ' ', result).strip()

        if result != original:
            logger.debug(f"[ResearchPlanner] Sanitized query: '{original}' -> '{result}'")

        return result if result else original

    def _fallback_plan(self, query: str) -> ResearchPlan:
        """Generate a comprehensive fallback plan if LLM plan generation fails."""
        return ResearchPlan(
            steps=[
                ResearchStepDef(
                    step_index=0,
                    title="Core Definitions and Formal Foundations",
                    description=f"Research formal definitions, theoretical foundations, and seminal works for: {query}",
                    search_queries=[
                        f"{query} survey arxiv 2024",
                        f"{query} formal definition research paper",
                        f"{query} seminal work foundational",
                        f"{query} overview technical explanation",
                    ],
                ),
                ResearchStepDef(
                    step_index=1,
                    title="Architecture and Technical Design",
                    description=f"Investigate architectures, algorithms, and design patterns for: {query}",
                    search_queries=[
                        f"{query} architecture design paper",
                        f"{query} algorithm technical details arxiv",
                        f"{query} implementation neural network",
                        f"{query} model design deep learning",
                    ],
                ),
                ResearchStepDef(
                    step_index=2,
                    title="Empirical Results and Benchmarks",
                    description=f"Find benchmark comparisons, evaluation results, and performance metrics for: {query}",
                    search_queries=[
                        f"{query} benchmark evaluation results 2024",
                        f"{query} SOTA state of the art performance",
                        f"{query} ablation study arxiv paper",
                        f"{query} COCO ImageNet benchmark comparison",
                    ],
                ),
                ResearchStepDef(
                    step_index=3,
                    title="Current Research Frontiers",
                    description=f"Latest developments, open problems, and active research for: {query}",
                    search_queries=[
                        f"{query} latest advances 2024 2025 arxiv",
                        f"{query} recent developments research",
                        f"{query} open problems challenges survey",
                        f"{query} future directions new paper",
                    ],
                ),
                ResearchStepDef(
                    step_index=4,
                    title="Applications and Case Studies",
                    description=f"Real-world applications, industry use cases, and practical implementations of: {query}",
                    search_queries=[
                        f"{query} real world applications deployment",
                        f"{query} industry use case practical",
                        f"{query} production system tutorial",
                        f"{query} case study results benchmark",
                    ],
                ),
            ],
            strategy="depth-first",
            estimated_sources=20,
            report_structure=[
                {"section_title": f"Foundations of {query}", "section_purpose": "Core definitions and theoretical underpinnings", "expected_content": "Definitions, formalisms, history"},
                {"section_title": "Technical Architecture", "section_purpose": "How the system/concept works at a technical level", "expected_content": "Algorithms, designs, mathematical formulations"},
                {"section_title": "Empirical Landscape", "section_purpose": "Benchmark results and evaluations", "expected_content": "Tables, metrics, comparisons"},
                {"section_title": "Frontiers and Open Problems", "section_purpose": "Where the field is heading", "expected_content": "Active research, unsolved challenges"},
            ],
        )

    def _parse_json_response(self, response: str):
        """Parse JSON from LLM response, handling common formatting issues."""
        if not response:
            return None

        text = response.strip()

        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try finding JSON within the text
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue

        return None