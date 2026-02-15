"""
Fragment Extractor — LLM-powered extraction of reusable memory fragments
from completed conversation turns.

Identifies:
- Knowledge fragments (factual information worth remembering)
- Solutions (problem → resolution patterns)
- Entity mentions (key proper nouns + context)
"""

import uuid
import json
from datetime import datetime
from typing import List, Optional
from loguru import logger

from backend.models.memory import MemoryFragment, MemoryFragmentType


# System prompt for the fragment extraction LLM call
EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction agent. Given a user query and the assistant's answer, extract reusable memory fragments.

Rules:
1. Only extract genuinely useful, self-contained information.
2. Each fragment must be understandable WITHOUT the original conversation context.
3. Maximum 3 fragments per turn to avoid bloat.
4. If the answer is too short (<200 chars) or trivial (greetings, acknowledgements), return an empty array.

Output ONLY a valid JSON array. Each item must have:
- "type": one of "knowledge", "solution", "preference", "entity"
- "content": the extracted fragment text (1-3 sentences, self-contained)
- "tags": array of 1-4 keyword tags

Example output:
[
  {"type": "knowledge", "content": "ArangoDB supports native graph traversal via AQL with GRAPH keyword and configurable min/max depth.", "tags": ["arangodb", "graph", "aql"]},
  {"type": "solution", "content": "To fix CORS errors in FastAPI, add CORSMiddleware with allow_origins=['*'] and allow_methods=['*'].", "tags": ["fastapi", "cors", "fix"]}
]

If nothing is worth extracting, return: []
"""


class FragmentExtractor:
    """
    Extracts structured memory fragments from completed conversation turns.
    Uses the existing LLMRouter for extraction.
    """

    def __init__(self, llm_router):
        """
        Args:
            llm_router: LLMRouter instance for making LLM calls
        """
        self.llm = llm_router

    def extract(
        self,
        query: str,
        answer: str,
        sources: List[str],
        session_id: str,
        namespace: str = "global"
    ) -> List[MemoryFragment]:
        """
        Extract memory fragments from a completed turn.
        
        Args:
            query: The user's original query
            answer: The assistant's synthesized answer
            sources: List of source documents used
            session_id: Current chat session ID
            namespace: Namespace to assign to extracted fragments
            
        Returns:
            List of MemoryFragment objects (max 3)
        """
        # Skip extraction for very short/trivial answers
        if not answer or len(answer.strip()) < 200:
            logger.debug("Answer too short for fragment extraction, skipping")
            return []

        try:
            user_message = (
                f"User Query: {query}\n\n"
                f"Assistant Answer: {answer[:2000]}\n\n"  # Cap to avoid context overflow
                f"Sources Used: {', '.join(sources[:5]) if sources else 'None'}"
            )

            response = self.llm.complete(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                max_tokens=800,
                temperature=0.3  # Low temp for structured extraction
            )

            if not response:
                return []

            # Parse JSON from response
            fragments = self._parse_response(response, session_id, namespace)
            return fragments

        except Exception as e:
            logger.error(f"Fragment extraction failed: {e}")
            return []

    def _parse_response(
        self,
        response: str,
        session_id: str,
        namespace: str
    ) -> List[MemoryFragment]:
        """Parse the LLM JSON response into MemoryFragment objects."""
        try:
            # Handle potential markdown code blocks
            text = response.strip()
            if text.startswith("```"):
                # Extract content between code fences
                lines = text.split("\n")
                text = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                )

            raw_fragments = json.loads(text)
            
            if not isinstance(raw_fragments, list):
                logger.warning("Fragment extraction returned non-list, ignoring")
                return []

            fragments = []
            now = datetime.utcnow().isoformat()

            for item in raw_fragments[:3]:  # Hard cap at 3
                frag_type = item.get("type", "knowledge")
                content = item.get("content", "").strip()
                tags = item.get("tags", [])

                if not content or len(content) < 20:
                    continue  # Skip empty/trivial fragments

                # Validate fragment type
                try:
                    fragment_type = MemoryFragmentType(frag_type)
                except ValueError:
                    fragment_type = MemoryFragmentType.KNOWLEDGE

                fragment = MemoryFragment(
                    fragment_id=str(uuid.uuid4()),
                    session_id=session_id,
                    fragment_type=fragment_type,
                    content=content,
                    tags=tags if isinstance(tags, list) else [],
                    namespace=namespace,
                    importance_score=0.5,  # Default, will be updated by MemoryScorer
                    created_at=now
                )
                fragments.append(fragment)

            logger.info(f"Extracted {len(fragments)} memory fragments")
            return fragments

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse fragment extraction JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Fragment parsing error: {e}")
            return []
