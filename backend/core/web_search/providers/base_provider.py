"""
Abstract base class for all web search providers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class SearchResult:
    """Standardized search result from any provider."""
    title: str
    url: str
    snippet: str
    source_provider: str
    date: Optional[str] = None
    relevance_score: float = 0.0
    provider_count: int = 1
    diversity_bonus: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0


class BaseSearchProvider(ABC):
    """Abstract base class for search providers."""
    
    def __init__(self, name: str, max_results: int = 5):
        self.name = name
        self.max_results = max_results
    
    @abstractmethod
    async def search(self, query: str) -> List[SearchResult]:
        """
        Execute a search query and return standardized results.
        Must be implemented by all providers.
        """
        pass
    
    def _clean_snippet(self, text: str, max_length: int = 500) -> str:
        """Clean and truncate a text snippet."""
        if not text:
            return ""
        # Remove excessive whitespace
        text = " ".join(text.split())
        if len(text) > max_length:
            text = text[:max_length] + "..."
        return text
