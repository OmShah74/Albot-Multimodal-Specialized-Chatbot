"""
Deep Web Content Scraper — Full page content extraction for deep research.
Goes beyond search engine snippets to extract and clean complete article text.
"""

import asyncio
import re
from typing import List, Optional
from urllib.parse import urlparse
from datetime import datetime
from loguru import logger

import httpx
from bs4 import BeautifulSoup

from backend.core.deep_research.models import ScrapedPage, ResearchConfig


# Try importing trafilatura for robust extraction, fall back gracefully
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    logger.warning("[WebScraper] trafilatura not installed, using BeautifulSoup fallback")

# Try importing readability as secondary fallback
try:
    from readability import Document as ReadabilityDocument
    HAS_READABILITY = True
except ImportError:
    HAS_READABILITY = False


class DeepWebScraper:
    """
    Asynchronous web scraper for deep research.
    
    Extracts full article/page text using a multi-tier extraction strategy:
    1. trafilatura (best quality for articles)
    2. readability-lxml (good for news/blog content)
    3. BeautifulSoup (raw fallback)
    
    Features:
    - Async batch scraping with concurrency control
    - Content length limiting
    - Timeout handling
    - Domain extraction
    """

    # Domains to skip (social media, login pages, etc.)
    SKIP_DOMAINS = {
        "twitter.com", "x.com", "facebook.com", "instagram.com",
        "tiktok.com", "linkedin.com", "pinterest.com",
        "youtube.com", "youtu.be",  # These need special handling
    }

    # Common headers to mimic a browser
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }

    def __init__(self, config: Optional[ResearchConfig] = None):
        self.config = config or ResearchConfig()
        self.timeout = self.config.scrape_timeout
        self.max_content = self.config.max_content_per_page
        self.max_concurrent = self.config.concurrent_scrapes

    async def scrape_urls(self, urls: List[str]) -> List[ScrapedPage]:
        """
        Scrape multiple URLs in parallel with concurrency control.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of ScrapedPage results (one per URL)
        """
        # Deduplicate and filter
        seen = set()
        unique_urls = []
        for url in urls:
            normalized = self._normalize_url(url)
            if normalized and normalized not in seen:
                seen.add(normalized)
                domain = urlparse(normalized).netloc.replace("www.", "")
                if domain not in self.SKIP_DOMAINS:
                    unique_urls.append(normalized)
        
        if not unique_urls:
            return []

        logger.info(f"[WebScraper] Scraping {len(unique_urls)} URLs (max concurrent: {self.max_concurrent})")
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def _bounded_scrape(url: str) -> ScrapedPage:
            async with semaphore:
                return await self._scrape_single(url)
        
        tasks = [_bounded_scrape(url) for url in unique_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        scraped = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"[WebScraper] Exception scraping {unique_urls[i]}: {result}")
                scraped.append(ScrapedPage(
                    url=unique_urls[i],
                    success=False,
                    error=str(result),
                    domain=urlparse(unique_urls[i]).netloc.replace("www.", ""),
                ))
            else:
                scraped.append(result)
        
        success_count = sum(1 for s in scraped if s.success)
        logger.info(f"[WebScraper] Completed: {success_count}/{len(scraped)} successful")
        
        return scraped

    async def _scrape_single(self, url: str) -> ScrapedPage:
        """Scrape a single URL with multi-tier content extraction."""
        domain = urlparse(url).netloc.replace("www.", "")
        
        try:
            async with httpx.AsyncClient(
                headers=self.DEFAULT_HEADERS,
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
                verify=False,  # Some research sites have bad certs
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                html = response.text
                
                if not html or len(html) < 100:
                    return ScrapedPage(url=url, domain=domain, success=False, error="Empty response")
                
                # Extract content using multi-tier strategy
                title, content = self._extract_content(html, url)
                
                if not content or len(content.strip()) < 50:
                    return ScrapedPage(url=url, domain=domain, title=title, success=False, error="No meaningful content extracted")
                
                # Trim to max length
                if len(content) > self.max_content:
                    content = content[:self.max_content] + "\n\n[Content truncated...]"
                
                word_count = len(content.split())
                
                return ScrapedPage(
                    url=url,
                    title=title,
                    content=content,
                    word_count=word_count,
                    domain=domain,
                    scraped_at=datetime.utcnow().isoformat(),
                    success=True,
                )
        
        except httpx.TimeoutException:
            return ScrapedPage(url=url, domain=domain, success=False, error="Timeout")
        except httpx.HTTPStatusError as e:
            return ScrapedPage(url=url, domain=domain, success=False, error=f"HTTP {e.response.status_code}")
        except Exception as e:
            return ScrapedPage(url=url, domain=domain, success=False, error=str(e)[:200])

    def _extract_content(self, html: str, url: str) -> tuple:
        """
        Multi-tier content extraction:
        1. trafilatura (best quality)
        2. readability-lxml (fallback)
        3. BeautifulSoup (raw fallback)
        
        Returns: (title, content)
        """
        title = ""
        content = ""
        
        # Tier 1: trafilatura
        if HAS_TRAFILATURA:
            try:
                extracted = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                    favor_recall=True,
                    url=url,
                )
                if extracted and len(extracted.strip()) > 100:
                    content = extracted
                    # Get title separately
                    metadata = trafilatura.extract_metadata(html)
                    if metadata and metadata.title:
                        title = metadata.title
            except Exception as e:
                logger.debug(f"[WebScraper] trafilatura failed for {url}: {e}")
        
        # Tier 2: readability-lxml
        if not content and HAS_READABILITY:
            try:
                doc = ReadabilityDocument(html)
                title = title or doc.short_title()
                summary_html = doc.summary()
                soup = BeautifulSoup(summary_html, "html.parser")
                content = soup.get_text(separator="\n", strip=True)
            except Exception as e:
                logger.debug(f"[WebScraper] readability failed for {url}: {e}")
        
        # Tier 3: BeautifulSoup raw extraction
        if not content:
            try:
                soup = BeautifulSoup(html, "html.parser")
                
                # Get title
                title_tag = soup.find("title")
                if title_tag:
                    title = title or title_tag.get_text(strip=True)
                
                # Remove noise elements
                for tag in soup(["script", "style", "nav", "footer", "header",
                                 "aside", "form", "iframe", "noscript"]):
                    tag.decompose()
                
                # Try article/main content first
                main_content = soup.find("article") or soup.find("main") or soup.find("div", class_=re.compile(r"content|article|post"))
                
                if main_content:
                    content = main_content.get_text(separator="\n", strip=True)
                else:
                    # Fall back to full body
                    body = soup.find("body")
                    if body:
                        content = body.get_text(separator="\n", strip=True)
            except Exception as e:
                logger.debug(f"[WebScraper] BeautifulSoup failed for {url}: {e}")
        
        # Clean up the content
        content = self._clean_text(content)
        
        return title, content

    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing excessive whitespace and noise."""
        if not text:
            return ""
        
        # Collapse multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Collapse multiple spaces
        text = re.sub(r'[ \t]{2,}', ' ', text)
        
        # Remove very short lines (likely nav items)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 3 or stripped == "":
                cleaned_lines.append(stripped)
        
        return '\n'.join(cleaned_lines).strip()

    def _normalize_url(self, url: str) -> Optional[str]:
        """Normalize a URL for deduplication."""
        if not url:
            return None
        
        url = url.strip()
        
        # Ensure scheme
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        # Remove trailing slash
        url = url.rstrip("/")
        
        # Remove common tracking parameters
        url = re.sub(r'[?&](utm_\w+|ref|source|fbclid|gclid)=[^&]*', '', url)
        
        # Clean up dangling ? or &
        url = url.rstrip("?&")
        
        return url
