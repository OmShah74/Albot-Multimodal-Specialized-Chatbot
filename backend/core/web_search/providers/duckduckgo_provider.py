"""
DuckDuckGo Search Provider - Uses the duckduckgo_search library with raw fallback.
Completely free, no API key required.
"""
import asyncio
from typing import List
from loguru import logger
import requests
from bs4 import BeautifulSoup

from backend.core.web_search.providers.base_provider import BaseSearchProvider, SearchResult


class DuckDuckGoProvider(BaseSearchProvider):
    """Search provider using DuckDuckGo with library and raw fallback."""
    
    def __init__(self, max_results: int = 5):
        super().__init__(name="DuckDuckGo", max_results=max_results)
    
    async def search(self, query: str) -> List[SearchResult]:
        """Execute search via DuckDuckGo with fallback backends."""
        backends = ['api', 'html', 'lite']
        loop = asyncio.get_event_loop()
        
        # 1. Try library backends
        for backend in backends:
            try:
                results = await loop.run_in_executor(
                    None, 
                    lambda: self._sync_search_lib(query, backend)
                )
                
                if results:
                    logger.info(f"DuckDuckGo ({backend}) returned {len(results)} results")
                    return results
                
            except Exception as e:
                logger.warning(f"DuckDuckGo library backend '{backend}' failed: {e}")
                continue
        
        # 2. Try Raw Scraping Fallback
        logger.warning("All DuckDuckGo backends failed. Attempting raw HTML fallback...")
        try:
            results = await loop.run_in_executor(
                    None, 
                    lambda: self._raw_html_fallback(query)
                )
            if results:
                logger.info(f"DuckDuckGo (Raw) returned {len(results)} results")
                return results
        except Exception as e:
            logger.error(f"DuckDuckGo Raw Fallback failed: {e}")

        logger.error("All DuckDuckGo attempts failed")
        return []

    def _sync_search_lib(self, query: str, backend: str) -> List[SearchResult]:
        """Synchronous DuckDuckGo search using library."""
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            # backend='api' is default and robust
            ddg_results = list(ddgs.text(query, max_results=self.max_results, backend=backend))
            
            for r in ddg_results:
                snippet = r.get("body", r.get("snippet", ""))
                if not snippet: continue
                
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", r.get("link", "")),
                    snippet=self._clean_snippet(snippet),
                    source_provider=f"DuckDuckGo ({backend})",
                    date=r.get("date", None)
                ))
        return results

    def _raw_html_fallback(self, query: str) -> List[SearchResult]:
        """Fallback: Scrape html.duckduckgo.com directly."""
        url = "https://html.duckduckgo.com/html/"
        params = {'q': query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Referer": "https://html.duckduckgo.com/"
        }
        
        results = []
        resp = requests.post(url, data=params, headers=headers, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Results are in div.result
        # The structure of html.duckduckgo.com is usually:
        # <div class="result"> <h2 class="result__title"><a ...></a></h2> <a class="result__snippet">...</a> </div>
        
        for div in soup.find_all('div', class_='result'):
            try:
                # Title & Link
                title_a = div.find('a', class_='result__a')
                if not title_a: continue
                
                title = title_a.get_text(strip=True)
                link = title_a['href']
                
                # Snippet
                snippet = ""
                snippet_a = div.find('a', class_='result__snippet')
                if snippet_a:
                    snippet = snippet_a.get_text(strip=True)
                
                if link and title:
                    results.append(SearchResult(
                        title=title,
                        url=link,
                        snippet=self._clean_snippet(snippet),
                        source_provider="DuckDuckGo (Raw)"
                    ))
                    if len(results) >= self.max_results:
                        break
            except Exception:
                continue
                
        return results
