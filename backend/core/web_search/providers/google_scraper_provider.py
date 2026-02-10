"""
Google Scraper Provider - Lightweight Google search via requests.
Completely free, no API key required. Uses Google's public search endpoint.
"""
import asyncio
import re
from typing import List
from loguru import logger
import requests
from urllib.parse import quote_plus, urlparse, parse_qs, unquote

from backend.core.web_search.providers.base_provider import BaseSearchProvider, SearchResult


class GoogleScraperProvider(BaseSearchProvider):
    """Search provider that scrapes Google search results."""
    
    SEARCH_URL = "https://www.google.com/search"
    
    def __init__(self, max_results: int = 5):
        super().__init__(name="Google", max_results=max_results)
    
    async def search(self, query: str) -> List[SearchResult]:
        """Execute search via Google scraping."""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._sync_search, query)
            logger.info(f"Google scraper returned {len(results)} results for: {query}")
            return results
        except Exception as e:
            logger.error(f"Google scraper failed: {e}")
            return []
    
    def _sync_search(self, query: str) -> List[SearchResult]:
        """Synchronous Google search via HTML parsing."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }
        
        params = {
            "q": query,
            "num": self.max_results + 2,  # Request a few extra in case some are filtered
            "hl": "en"
        }
        
        try:
            resp = requests.get(
                self.SEARCH_URL,
                params=params,
                headers=headers,
                timeout=10
            )
            resp.raise_for_status()
            html = resp.text
            return self._parse_results(html)
        except Exception as e:
            logger.error(f"Google request failed: {e}")
            return []
    
    def _parse_results(self, html: str) -> List[SearchResult]:
        """Parse search results from Google HTML response."""
        results = []
        
        # Extract result blocks using regex patterns
        # Look for link + title + snippet patterns
        # Pattern to find search result divs
        link_pattern = re.compile(
            r'<a\s+href="/url\?q=([^"&]+)[^"]*"[^>]*>(.*?)</a>',
            re.DOTALL
        )
        
        matches = link_pattern.findall(html)
        
        seen_urls = set()
        for url_encoded, title_html in matches:
            url = unquote(url_encoded)
            
            # Skip internal Google links
            if "google.com" in url or "youtube.com/sorry" in url:
                continue
            if url.startswith("/"):
                continue
            if url in seen_urls:
                continue
            
            seen_urls.add(url)
            
            # Clean title
            title = re.sub(r'<[^>]+>', '', title_html).strip()
            if not title or len(title) < 3:
                continue
            
            # Try to extract snippet - look for text near the URL
            snippet = self._extract_snippet(html, url)
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=self._clean_snippet(snippet),
                source_provider=self.name
            ))
            
            if len(results) >= self.max_results:
                break
        
        return results
    
    def _extract_snippet(self, html: str, url: str) -> str:
        """Try to extract a snippet text near a URL in the HTML."""
        # Find text blocks near the URL reference
        escaped_url = re.escape(url[:50])  # Use first 50 chars of URL
        
        # Look for spans with class containing snippet-like content
        snippet_patterns = [
            # Common Google snippet patterns
            re.compile(r'<span[^>]*>([^<]{40,300})</span>', re.DOTALL),
        ]
        
        # Find the URL position and get surrounding text
        url_pos = html.find(url[:30])
        if url_pos > 0:
            # Search in a window around the URL
            window = html[url_pos:url_pos + 2000]
            for pattern in snippet_patterns:
                match = pattern.search(window)
                if match:
                    snippet = match.group(1)
                    # Remove HTML tags
                    snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                    if len(snippet) > 30:
                        return snippet
        
        return ""
