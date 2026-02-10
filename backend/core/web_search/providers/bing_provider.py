"""
Bing Search Provider - Scrapes Bing search results.
Completely free, no API key required.
"""
import asyncio
import re
from typing import List
from loguru import logger
import requests
from urllib.parse import unquote

from backend.core.web_search.providers.base_provider import BaseSearchProvider, SearchResult


class BingSearchProvider(BaseSearchProvider):
    """Search provider that scrapes Bing search results."""
    
    SEARCH_URL = "https://www.bing.com/search"
    
    def __init__(self, max_results: int = 5):
        super().__init__(name="Bing", max_results=max_results)
    
    async def search(self, query: str) -> List[SearchResult]:
        """Execute search via Bing scraping."""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._sync_search, query)
            logger.info(f"Bing scraper returned {len(results)} results for: {query}")
            return results
        except Exception as e:
            logger.error(f"Bing scraper failed: {e}")
            return []
    
    def _sync_search(self, query: str) -> List[SearchResult]:
        """Synchronous Bing search via HTML parsing."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.bing.com/",
            "Cookie": "SRCHHPGUSR=SRCHLANG=en",  # Force English
        }
        
        params = {
            "q": query,
            "count": self.max_results + 5,
            "form": "QBLH"
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
            logger.error(f"Bing request failed: {e}")
            return []
    
    def _parse_results(self, html: str) -> List[SearchResult]:
        """Parse search results from Bing HTML response using BeautifulSoup."""
        from bs4 import BeautifulSoup
        results = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Bing result blocks usually have class 'b_algo'
            blocks = soup.find_all('li', class_='b_algo')
            
            for block in blocks:
                try:
                    # Title and URL
                    h2 = block.find('h2')
                    if not h2:
                        continue
                        
                    a_tag = h2.find('a')
                    if not a_tag:
                        continue
                        
                    url = a_tag.get('href')
                    title = a_tag.get_text(strip=True)
                    
                    if not url or not url.startswith('http'):
                        continue
                    
                    # Snippet extraction
                    snippet = ""
                    # Selector 1: <div class="b_caption"><p>
                    caption_div = block.find('div', class_='b_caption')
                    if caption_div:
                        p_tag = caption_div.find('p')
                        if p_tag:
                            snippet = p_tag.get_text(strip=True)
                    
                    # Selector 2: <p> directly in block (less common but possible)
                    if not snippet:
                        p_tag = block.find('p')
                        if p_tag:
                            snippet = p_tag.get_text(strip=True)
                            
                    # Selector 3: Generic div with text if no p tag
                    if not snippet and caption_div:
                        snippet = caption_div.get_text(strip=True)

                    if snippet:
                        snippet = self._clean_snippet(snippet)
                    
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source_provider=self.name
                    ))
                    
                    if len(results) >= self.max_results:
                        break
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
             logger.error(f"Bing parsing failed: {e}")
        
        return results
