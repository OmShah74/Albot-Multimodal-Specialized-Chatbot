"""
Wikipedia Search Provider - Uses the Wikipedia API for encyclopedic knowledge.
Completely free, no API key required.
"""
import asyncio
from typing import List
from loguru import logger
import requests

from backend.core.web_search.providers.base_provider import BaseSearchProvider, SearchResult


class WikipediaProvider(BaseSearchProvider):
    """Search provider using Wikipedia's REST API."""
    
    SEARCH_URL = "https://en.wikipedia.org/w/api.php"
    
    def __init__(self, max_results: int = 3):
        super().__init__(name="Wikipedia", max_results=max_results)
    
    async def search(self, query: str) -> List[SearchResult]:
        """Execute search via Wikipedia API."""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._sync_search, query)
            logger.info(f"Wikipedia returned {len(results)} results for: {query}")
            return results
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return []
    
    def _sync_search(self, query: str) -> List[SearchResult]:
        """Synchronous Wikipedia search."""
        results = []
        
        # Step 1: Search for matching articles
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": self.max_results,
            "format": "json",
            "srprop": "snippet|timestamp",
            "utf8": 1
        }
        
        headers = {
            "User-Agent": "Albot/1.0 (Educational AI Assistant)"
        }
        
        try:
            resp = requests.get(self.SEARCH_URL, params=search_params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            search_results = data.get("query", {}).get("search", [])
            
            for item in search_results:
                title = item.get("title", "")
                # Clean HTML tags from snippet
                snippet = item.get("snippet", "")
                snippet = snippet.replace('<span class="searchmatch">', "").replace("</span>", "")
                snippet = self._clean_snippet(snippet)
                
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                
                results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source_provider=self.name,
                    date=item.get("timestamp", None)
                ))
            
        except Exception as e:
            logger.error(f"Wikipedia API error: {e}")
        
        return results
