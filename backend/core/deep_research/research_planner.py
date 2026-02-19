"""
Agentic Research Planner — Uses LLM to generate, adapt, and evaluate research plans.
Implements the PLAN phase of the deep research pipeline.
"""

import json
from typing import List, Optional
from loguru import logger

from backend.core.deep_research.models import (
    ResearchPlan, ResearchStepDef, ResearchConfig
)


class ResearchPlanner:
    """
    Generates and adapts multi-step research plans using the LLM.
    
    The planner:
    1. Analyzes the user's research query
    2. Generates an ordered plan of 10-12 research steps
    3. Creates targeted search queries for each step
    4. Can adapt the plan mid-research based on findings
    5. Evaluates whether enough information has been gathered
    """

    def __init__(self, llm_router, config: Optional[ResearchConfig] = None):
        self.llm = llm_router
        self.config = config or ResearchConfig()

    async def generate_plan(self, query: str) -> ResearchPlan:
        """
        Generate a research plan with ordered steps and search queries.
        
        Args:
            query: The user's research query
            
        Returns:
            ResearchPlan with steps and search queries
        """
        max_steps = self.config.max_steps
        
        system_prompt = f"""You are a research planning expert. Given a research query, create a comprehensive research plan.

Rules:
- Create between 8 and 12 focused research steps
- Each step should have a clear title, description, and 2-4 specific web search queries 
- Steps should be logically ordered (background → specifics → analysis → synthesis)
- Search queries should be specific and diverse (don't repeat the same query)
- Respond ONLY with valid JSON, no markdown or extra text

Output format:
{{
  "strategy": "breadth-first",
  "estimated_sources": <number>,
  "steps": [
    {{
      "step_index": 0,
      "title": "Step title",
      "description": "What this step investigates",
      "search_queries": ["query 1", "query 2", "query 3"]
    }}
  ]
}}"""

        user_prompt = f"Create a detailed research plan for this query:\n\n\"{query}\""

        try:
            response = self.llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                max_tokens=2000,
                temperature=0.4,
            )
            
            plan_data = self._parse_json_response(response)
            
            if not plan_data or "steps" not in plan_data:
                logger.warning("[ResearchPlanner] Failed to parse plan, using fallback")
                return self._fallback_plan(query)
            
            steps = []
            for i, step_data in enumerate(plan_data["steps"][:max_steps]):
                steps.append(ResearchStepDef(
                    step_index=i,
                    title=step_data.get("title", f"Research Step {i+1}"),
                    description=step_data.get("description", ""),
                    search_queries=step_data.get("search_queries", [query]),
                    status="pending",
                ))
            
            plan = ResearchPlan(
                steps=steps,
                strategy=plan_data.get("strategy", "breadth-first"),
                estimated_sources=plan_data.get("estimated_sources", len(steps) * 3),
            )
            
            logger.info(f"[ResearchPlanner] Generated plan with {len(steps)} steps")
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
        Used for recursive expansion of the research.
        
        Args:
            step_title: Current step being researched
            findings_summary: Summary of findings so far
            original_query: The original research query
            
        Returns:
            List of follow-up search queries
        """
        system_prompt = """You are a research assistant. Based on the current research findings, generate 2-3 follow-up search queries that:
1. Fill gaps in the current understanding
2. Explore interesting angles discovered in the findings while not deviating from the user's problem statement domain
3. Look for contradicting or supporting evidence

Respond ONLY with a JSON array of query strings, e.g.: ["query 1", "query 2"]"""

        user_prompt = f"""Original research question: "{original_query}"
Current step: "{step_title}"

Findings so far:
{findings_summary[:2000]}

Generate follow-up search queries:"""

        try:
            response = self.llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.5,
            )
            
            queries = self._parse_json_response(response)
            if isinstance(queries, list):
                return [str(q) for q in queries[:4]]
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
        
        Returns:
            True if research is sufficient, False if more is needed
        """
        system_prompt = """You are a research evaluator. Determine if enough information has been gathered to comprehensively answer the research question.

Consider:
- Coverage: Are all major aspects of the topic covered?
- Depth: Is there enough detail for each aspect?
- Sources: Is the information from diverse and reliable sources?

Respond with ONLY "yes" or "no"."""

        user_prompt = f"""Research question: "{query}"
Sources consulted: {sources_count}

Summary of findings:
{findings_summary[:3000]}

Is the research sufficient? (yes/no)"""

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

    def _fallback_plan(self, query: str) -> ResearchPlan:
        """Generate a simple fallback plan if LLM plan generation fails."""
        return ResearchPlan(
            steps=[
                ResearchStepDef(
                    step_index=0,
                    title="Background Research",
                    description=f"Research background information on: {query}",
                    search_queries=[query, f"{query} overview", f"{query} introduction"],
                ),
                ResearchStepDef(
                    step_index=1,
                    title="Detailed Analysis",
                    description=f"Deep dive into key aspects of: {query}",
                    search_queries=[f"{query} detailed analysis", f"{query} key findings", f"{query} research"],
                ),
                ResearchStepDef(
                    step_index=2,
                    title="Current Developments",
                    description=f"Latest developments and trends in: {query}",
                    search_queries=[f"{query} latest developments 2025", f"{query} recent advances", f"{query} trends"],
                ),
            ],
            strategy="breadth-first",
            estimated_sources=9,
        )

    def _parse_json_response(self, response: str):
        """Parse JSON from LLM response, handling common formatting issues."""
        if not response:
            return None
        
        text = response.strip()
        
        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (code fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try finding JSON within the text
        # Look for { ... } or [ ... ]
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
        
        return None
