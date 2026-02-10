"""
SearXNG Search Provider - Queries public SearXNG instances.
Completely free, no API key required. Uses public instances.
"""
import asyncio
import random
from typing import List
from loguru import logger
import requests

from backend.core.web_search.providers.base_provider import BaseSearchProvider, SearchResult


class SearXNGProvider(BaseSearchProvider):
    """Search provider using public SearXNG instances."""
    
    # List of public SearXNG instances with JSON API support
    # Updated with larger pool for reliability
    PUBLIC_INSTANCES = [
        "https://searx.be",
        "https://search.mdosch.de",
        "https://searx.space",
        "https://search.ononoki.org",
        "https://searx.work",
        "https://northboot.xyz",
        "https://search.rhscz.eu",
        "https://searx.webheberg.ini",
        "https://search.sapti.me",
        "https://search.gcomm.ch",
        "https://searx.name",
        "https://opnxng.com",
    ]
    
    def __init__(self, max_results: int = 5, instance_url: str = None):
        super().__init__(name="SearXNG", max_results=max_results)
        self.instance_url = instance_url
    
    async def search(self, query: str) -> List[SearchResult]:
        """Execute search via SearXNG."""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._sync_search, query)
            logger.info(f"SearXNG returned {len(results)} results for: {query}")
            return results
        except Exception as e:
            logger.error(f"SearXNG search failed: {e}")
            return []
    
    def _sync_search(self, query: str) -> List[SearchResult]:
        """Synchronous SearXNG search with instance fallback."""
        # Try instances in random order for load balancing
        instances = [self.instance_url] if self.instance_url else list(self.PUBLIC_INSTANCES)
        random.shuffle(instances)
        
        for instance in instances:
            try:
                results = self._query_instance(instance, query)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"SearXNG instance {instance} failed: {e}")
                continue
        
        logger.error("All SearXNG instances failed")
        return []
    
    def _query_instance(self, instance_url: str, query: str) -> List[SearchResult]:
        """Query a specific SearXNG instance."""
        params = {
            "q": query,
            "format": "json",
            "categories": "general",
            "language": "en",
            "pageno": 1
        }
        
        headers = {
            "User-Agent": "Albot/1.0 (Multimodal RAG Engine)",
            "Accept": "application/json"
        }
        
        resp = requests.get(
            f"{instance_url}/search",
            params=params,
            headers=headers,
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for item in data.get("results", [])[:self.max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=self._clean_snippet(item.get("content", "")),
                source_provider=self.name,
                date=item.get("publishedDate", None)
            ))
        
        return results
