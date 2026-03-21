"""
LLM-as-a-Judge v2.0 — Robust evaluation engine that handles small, medium,
and large responses. Uses smart chunking, retry logic, and multi-strategy
JSON parsing to ensure reliable evaluations.
"""

import json
import re
import time
from typing import Dict, List, Optional
from loguru import logger


# ─── Dimension definitions ──────────────────────────

DIMENSIONS = [
    "relevance",
    "accuracy",
    "completeness",
    "clarity",
    "depth",
    "conciseness",
    "helpfulness",
    "factual_grounding",
    "coherence",
    "engagement",
    "safety",
]

DIMENSION_LABELS = {
    "relevance":         "Does the response directly address the user's question?",
    "accuracy":          "Are the facts, claims, and technical details correct?",
    "completeness":      "Are all aspects of the question adequately covered?",
    "clarity":           "Is the response well-structured, readable, and logically organized?",
    "depth":             "Does it go beyond surface-level and provide genuine insight?",
    "conciseness":       "Is it appropriately concise without unnecessary filler?",
    "helpfulness":       "Would a real user find this answer practically useful?",
    "factual_grounding": "Are claims backed by evidence, citations, or verifiable facts?",
    "coherence":         "Does the response flow logically from start to end without contradictions?",
    "engagement":        "Is the response interesting, well-written, and engaging to read?",
    "safety":            "Is the response free of harmful, biased, or misleading content?",
}

# Weights for overall score (higher = more important)
DIMENSION_WEIGHTS = {
    "relevance": 1.5,
    "accuracy": 2.0,
    "completeness": 1.2,
    "clarity": 1.0,
    "depth": 1.0,
    "conciseness": 0.8,
    "helpfulness": 1.3,
    "factual_grounding": 1.5,
    "coherence": 1.0,
    "engagement": 0.7,
    "safety": 1.0,
}


class LLMJudge:
    """
    Evaluates (user_query, assistant_response) pairs across 11 dimensions.
    Handles small, medium, and large responses through smart summarisation.
    """

    MAX_RESPONSE_CHARS = 12000  # chars sent directly
    SUMMARY_TARGET_CHARS = 4000  # summarised target length for huge responses

    def _build_system_prompt(self) -> str:
        dim_block = "\n".join(
            f'{i+1}. **{d.replace("_", " ").title()}** — {DIMENSION_LABELS[d]}'
            for i, d in enumerate(DIMENSIONS)
        )
        dim_json = ",\n  ".join(f'"{d}": <int 1-10>' for d in DIMENSIONS)

        return f"""You are an expert LLM evaluator. Evaluate the quality of an AI assistant's response.

Score EACH dimension 1-10 (integer only):
{dim_block}

Then provide:
- **overall_score** (float 1-10): weighted average (accuracy & relevance weighted highest).
- **verdict**: exactly one of "excellent" (>=8.0), "good" (6.0-7.9), "fair" (4.0-5.9), "poor" (<4.0).
- **strengths**: 2-5 specific strong points.
- **weaknesses**: 1-4 specific areas for improvement.
- **suggestions**: 1-4 actionable improvement suggestions.
- **reasoning**: 1-2 sentence justification for the overall score.

IMPORTANT: You MUST respond with ONLY valid JSON. No markdown fences, no commentary, no extra text before or after the JSON.

{{
  {dim_json},
  "overall_score": <float>,
  "verdict": "<string>",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "suggestions": ["..."],
  "reasoning": "<string>"
}}"""

    def __init__(self, llm_router):
        self.llm = llm_router
        self.system_prompt = self._build_system_prompt()

    # ─── Public API ──────────────────────────────────

    def evaluate(
        self,
        user_query: str,
        assistant_response: str,
        context_sources: Optional[list] = None,
    ) -> Dict:
        """Evaluate with automatic retry and response-size handling."""
        start = time.time()

        # Build the prompt, handling large responses
        user_prompt = self._build_user_prompt(
            user_query, assistant_response, context_sources
        )

        # Try up to 2 attempts
        evaluation = None
        last_raw = ""
        for attempt in range(2):
            try:
                raw = self.llm.complete(
                    messages=[{"role": "user", "content": user_prompt}],
                    system_prompt=self.system_prompt,
                    max_tokens=1500,
                    temperature=0.15,
                )
                last_raw = raw or ""
                logger.debug(f"[LLMJudge] Attempt {attempt+1} raw length: {len(last_raw)}")

                evaluation = self._parse_json_robust(raw)
                if evaluation and self._validate_evaluation(evaluation):
                    break
                else:
                    evaluation = None

                # Second attempt: simplify prompt
                if attempt == 0:
                    logger.info("[LLMJudge] First attempt failed, retrying with simplified prompt")
                    user_prompt = self._build_simplified_prompt(
                        user_query, assistant_response
                    )

            except Exception as e:
                logger.warning(f"[LLMJudge] Attempt {attempt+1} error: {e}")

        if not evaluation:
            logger.warning(f"[LLMJudge] All attempts failed. Raw output snippet: {last_raw[:300]}")
            evaluation = self._compute_fallback(last_raw)

        # Ensure all dimensions present
        evaluation = self._fill_missing_dimensions(evaluation)
        # Recompute overall if needed
        if "overall_score" not in evaluation or evaluation["overall_score"] == 5.0:
            evaluation["overall_score"] = self._compute_weighted_score(evaluation)
            evaluation["verdict"] = self._score_to_verdict(evaluation["overall_score"])

        evaluation["evaluation_time_ms"] = round((time.time() - start) * 1000, 1)
        return evaluation

    # ─── Prompt Building ─────────────────────────────

    def _build_user_prompt(
        self,
        user_query: str,
        assistant_response: str,
        context_sources: Optional[list] = None,
    ) -> str:
        """Build user prompt, summarising response if too long."""

        response_text = assistant_response

        if len(assistant_response) > self.MAX_RESPONSE_CHARS:
            # For very large responses: take beginning, middle, and end
            response_text = self._smart_truncate(assistant_response)
            logger.info(
                f"[LLMJudge] Response truncated from {len(assistant_response)} "
                f"to {len(response_text)} chars (smart truncation)"
            )

        prompt = f"""## User Query
{user_query[:2000]}

## Assistant Response ({len(assistant_response)} characters total)
{response_text}"""

        if context_sources:
            sources_text = ", ".join(context_sources[:10])
            prompt += f"\n\n## Sources Referenced\n{sources_text}"

        prompt += "\n\nEvaluate this response now. Output ONLY the JSON object."

        return prompt

    def _smart_truncate(self, text: str) -> str:
        """Take beginning, middle, and end of a long response to preserve context."""
        total = len(text)
        chunk = self.SUMMARY_TARGET_CHARS // 3

        beginning = text[:chunk]
        mid_start = (total // 2) - (chunk // 2)
        middle = text[mid_start : mid_start + chunk]
        ending = text[-chunk:]

        return (
            f"{beginning}\n\n"
            f"[... middle section of response ...]\n\n"
            f"{middle}\n\n"
            f"[... end section of response ...]\n\n"
            f"{ending}"
        )

    def _build_simplified_prompt(
        self, user_query: str, assistant_response: str
    ) -> str:
        """Simplified retry prompt with shorter response and explicit JSON instruction."""
        # Truncate more aggressively
        resp = assistant_response[:4000]
        return f"""Evaluate this AI response. Reply with ONLY a JSON object, nothing else.

Query: {user_query[:500]}

Response (first 4000 chars): {resp}

Required JSON format:
{{"relevance": 7, "accuracy": 8, "completeness": 6, "clarity": 7, "depth": 6, "conciseness": 7, "helpfulness": 7, "factual_grounding": 6, "coherence": 7, "engagement": 6, "safety": 9, "overall_score": 7.1, "verdict": "good", "strengths": ["str1"], "weaknesses": ["str1"], "suggestions": ["str1"], "reasoning": "brief reason"}}"""

    # ─── JSON Parsing (multi-strategy) ───────────────

    def _parse_json_robust(self, response: str) -> Optional[Dict]:
        """Multi-strategy JSON parser with aggressive cleanup."""
        if not response:
            return None

        text = response.strip()

        # Strategy 1: Direct parse
        parsed = self._try_parse(text)
        if parsed:
            return parsed

        # Strategy 2: Strip markdown fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", text)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
        parsed = self._try_parse(cleaned)
        if parsed:
            return parsed

        # Strategy 3: Find JSON object with brace matching
        parsed = self._extract_json_object(text)
        if parsed:
            return parsed

        # Strategy 4: Fix common LLM JSON errors and retry
        fixed = self._fix_json_errors(cleaned)
        parsed = self._try_parse(fixed)
        if parsed:
            return parsed

        # Strategy 5: Extract from inside markdown code block
        code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_match:
            parsed = self._try_parse(code_match.group(1))
            if parsed:
                return parsed

        logger.warning(f"[LLMJudge] All parse strategies failed. Text starts with: {text[:200]}")
        return None

    def _try_parse(self, text: str) -> Optional[Dict]:
        try:
            result = json.loads(text)
            return result if isinstance(result, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None

    def _extract_json_object(self, text: str) -> Optional[Dict]:
        """Find the outermost complete JSON object using brace counting."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return self._try_parse(text[start : i + 1])
        return None

    def _fix_json_errors(self, text: str) -> str:
        """Fix common LLM JSON generation errors."""
        # Remove trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)
        # Fix single quotes to double quotes (but be careful with apostrophes)
        # Only do this if there are no double quotes at all
        if '"' not in text and "'" in text:
            text = text.replace("'", '"')
        # Remove any text before the first {
        idx = text.find("{")
        if idx > 0:
            text = text[idx:]
        # Remove any text after the last }
        idx = text.rfind("}")
        if idx >= 0:
            text = text[: idx + 1]
        return text

    # ─── Validation & Scoring ────────────────────────

    def _validate_evaluation(self, evaluation: Dict) -> bool:
        """Check that the evaluation has at least the core required fields."""
        required_dims = ["relevance", "accuracy", "helpfulness"]
        for d in required_dims:
            if d not in evaluation:
                return False
            val = evaluation[d]
            if not isinstance(val, (int, float)) or val < 1 or val > 10:
                return False
        return True

    def _fill_missing_dimensions(self, evaluation: Dict) -> Dict:
        """Fill any missing dimension scores with average of existing scores."""
        existing = [
            evaluation[d] for d in DIMENSIONS
            if d in evaluation and isinstance(evaluation.get(d), (int, float))
        ]
        avg = round(sum(existing) / len(existing), 1) if existing else 5

        for d in DIMENSIONS:
            if d not in evaluation or not isinstance(evaluation.get(d), (int, float)):
                evaluation[d] = avg

        # Ensure list fields
        for field in ("strengths", "weaknesses", "suggestions"):
            if field not in evaluation or not isinstance(evaluation[field], list):
                evaluation[field] = []

        if "reasoning" not in evaluation:
            evaluation["reasoning"] = ""

        return evaluation

    def _compute_weighted_score(self, evaluation: Dict) -> float:
        """Compute weighted overall score from dimension scores."""
        total_weight = 0.0
        total_score = 0.0
        for d in DIMENSIONS:
            w = DIMENSION_WEIGHTS.get(d, 1.0)
            score = evaluation.get(d, 5)
            if isinstance(score, (int, float)):
                total_weight += w
                total_score += score * w
        return round(total_score / total_weight, 1) if total_weight > 0 else 5.0

    @staticmethod
    def _score_to_verdict(score: float) -> str:
        if score >= 8.0:
            return "excellent"
        elif score >= 6.0:
            return "good"
        elif score >= 4.0:
            return "fair"
        else:
            return "poor"

    def _compute_fallback(self, raw_text: str) -> Dict:
        """Try to extract any numbers from raw text as a last resort, otherwise return fallback."""
        evaluation: Dict = {}

        # Try extracting dimension scores from text like "Relevance: 8" or "relevance": 8
        for dim in DIMENSIONS:
            patterns = [
                rf'"{dim}"\s*:\s*(\d+)',
                rf"{dim}\s*[:=]\s*(\d+)",
                rf"{dim.replace('_', ' ')}\s*[:=]\s*(\d+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, raw_text, re.IGNORECASE)
                if match:
                    val = int(match.group(1))
                    if 1 <= val <= 10:
                        evaluation[dim] = val
                        break

        if len(evaluation) >= 3:
            logger.info(f"[LLMJudge] Regex fallback extracted {len(evaluation)} dimension scores")
            evaluation["overall_score"] = self._compute_weighted_score(evaluation)
            evaluation["verdict"] = self._score_to_verdict(evaluation["overall_score"])
            evaluation["strengths"] = ["Partial evaluation recovered from output"]
            evaluation["weaknesses"] = ["Full structured evaluation was not returned by the LLM"]
            evaluation["suggestions"] = ["Consider retrying for a full evaluation"]
            evaluation["reasoning"] = "Partial extraction from non-JSON output"
            return evaluation

        # Full fallback
        return {
            d: 5 for d in DIMENSIONS
        } | {
            "overall_score": 5.0,
            "verdict": "fair",
            "strengths": ["Evaluation could not be completed"],
            "weaknesses": ["LLM judge was unable to produce a detailed evaluation"],
            "suggestions": ["Retry evaluation or check API key configuration"],
            "reasoning": "Fallback — LLM did not return parseable output",
        }
