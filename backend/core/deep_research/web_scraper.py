"""
Deep Web Content Scraper — Full page content extraction for deep research.
Goes beyond search engine snippets to extract and clean complete article text.
Supports arXiv PDF extraction with automatic cleanup.
"""

import asyncio
import os
import re
import tempfile
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

# Try importing PyMuPDF for PDF extraction
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("[WebScraper] PyMuPDF (fitz) not installed, PDF extraction disabled")


class DeepWebScraper:
    """
    Asynchronous web scraper for deep research.

    Extracts full article/page text using a multi-tier extraction strategy:
    1. trafilatura (best quality for articles)
    2. readability-lxml (good for news/blog content)
    3. BeautifulSoup (raw fallback)
    4. PyMuPDF for PDF content (arXiv, research papers)

    Features:
    - Async batch scraping with concurrency control
    - arXiv PDF download + text extraction + auto-cleanup
    - Blocked domain filtering (Wikipedia, unauthorized sources)
    - Content length limiting
    - Timeout handling
    - Domain extraction
    """

    # ─── Blocked Domains ─────────────────────────────────
    # These are skipped entirely — social media, wikis, and low-quality sites
    # ─── Blocked Domains ─────────────────────────────────
    # These are skipped entirely — social media, wikis, and low-quality sites
    BLOCKED_DOMAINS = {
        # Social media
        "twitter.com", "x.com", "facebook.com", "instagram.com",
        "tiktok.com", "linkedin.com", "pinterest.com",
        "youtube.com", "youtu.be",
        # Wikipedia and wiki sites (unreliable for deep research)
        "wikipedia.org", "en.wikipedia.org", "en.m.wikipedia.org",
        "wikibooks.org", "wikiversity.org", "wikiquote.org",
        "wiktionary.org", "wikisource.org", "wikimedia.org",
        "simple.wikipedia.org",
        # Low-quality / SEO-farm / aggregator sites
        "quora.com", "answers.com", "ehow.com",
        "wikihow.com", "about.com",
        "scribd.com",  # paywall
        "slideshare.net",
        "reddit.com",  # noisy forum content
        # Chinese Q&A / non-English platforms (irrelevant for English research)
        "zhihu.com",
        "baidu.com",
        "weibo.com",
        "csdn.net",
        # General news sites (not research sources)
        "ndtv.com",
        "timesofindia.com",
        "hindustantimes.com",
        "indiatoday.in",
        "thehindu.com",
        "news18.com",
        "indianexpress.com",
    }

    # ─── Prioritized Domains ─────────────────────────────
    # These get a boost in scrape ordering
    PRIORITY_DOMAINS = {
        "arxiv.org", "medium.com", "towardsdatascience.com",
        "openreview.net", "aclanthology.org",
        "semanticscholar.org", "scholar.google.com",
        "ieee.org", "acm.org", "springer.com", "nature.com",
        "sciencedirect.com", "pnas.org", "pubmed.ncbi.nlm.nih.gov",
        "distill.pub", "paperswithcode.com",
        "huggingface.co", "blog.research.google",
        "ai.meta.com", "deepmind.google",
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
        self._temp_files: List[str] = []  # Track temp PDF files for cleanup

    async def scrape_urls(self, urls: List[str]) -> List[ScrapedPage]:
        """
        Scrape multiple URLs in parallel with concurrency control.
        Automatically prioritizes research-quality domains.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of ScrapedPage results (one per URL)
        """
        # Deduplicate and filter blocked domains
        seen = set()
        unique_urls = []
        for url in urls:
            normalized = self._normalize_url(url)
            if normalized and normalized not in seen:
                seen.add(normalized)
                domain = urlparse(normalized).netloc.replace("www.", "")
                # Check if domain is blocked
                if self._is_blocked(domain):
                    logger.info(f"[WebScraper] Blocked domain skipped: {domain}")
                    continue
                unique_urls.append(normalized)

        if not unique_urls:
            return []

        # Sort: priority domains first
        unique_urls.sort(key=lambda u: self._priority_score(u), reverse=True)

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

    def cleanup_temp_files(self):
        """Delete all temporary PDF files downloaded during this session."""
        for fpath in self._temp_files:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
                    logger.debug(f"[WebScraper] Cleaned up temp file: {fpath}")
            except Exception as e:
                logger.warning(f"[WebScraper] Failed to clean temp file {fpath}: {e}")
        self._temp_files.clear()

    async def _scrape_single(self, url: str) -> ScrapedPage:
        """Scrape a single URL with multi-tier content extraction."""
        domain = urlparse(url).netloc.replace("www.", "")

        # ── Special handling: arXiv PDF ──
        if self._is_arxiv_pdf(url):
            return await self._scrape_arxiv_pdf(url, domain)

        # ── Special handling: arXiv abstract page → convert to PDF ──
        if "arxiv.org/abs/" in url:
            pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
            return await self._scrape_arxiv_pdf(pdf_url, domain)

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

    # ═══════════════════════════════════════════════════
    # arXiv / PDF Extraction
    # ═══════════════════════════════════════════════════

    async def _scrape_arxiv_pdf(self, pdf_url: str, domain: str) -> ScrapedPage:
        """
        Download an arXiv PDF, extract full text with PyMuPDF, then cleanup.
        Falls back to abstract-page scrape if PyMuPDF is unavailable.
        """
        if not HAS_PYMUPDF:
            # Fallback: try scraping the abstract HTML page instead
            abs_url = pdf_url.replace("/pdf/", "/abs/").replace(".pdf", "")
            logger.info(f"[WebScraper] PyMuPDF not available, falling back to abstract page: {abs_url}")
            return await self._scrape_single(abs_url)

        tmp_path = None
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                follow_redirects=True,
                verify=False,
            ) as client:
                response = await client.get(pdf_url)
                response.raise_for_status()

                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name
                    self._temp_files.append(tmp_path)

            # Extract text with PyMuPDF
            doc = fitz.open(tmp_path)
            title = doc.metadata.get("title", "") or ""
            pages_text = []
            for page in doc:
                pages_text.append(page.get_text("text"))
            doc.close()

            full_text = "\n\n".join(pages_text)

            # Clean up the temp file immediately after extraction
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                self._temp_files = [f for f in self._temp_files if f != tmp_path]
                logger.debug(f"[WebScraper] Cleaned up arXiv PDF: {tmp_path}")

            if not full_text or len(full_text.strip()) < 100:
                return ScrapedPage(url=pdf_url, domain=domain, success=False, error="PDF contained no extractable text")

            # Clean up extracted text
            full_text = self._clean_pdf_text(full_text)

            # Trim to max length
            if len(full_text) > self.max_content:
                full_text = full_text[:self.max_content] + "\n\n[Content truncated...]"

            # Extract arXiv ID for a better title
            if not title:
                arxiv_match = re.search(r'(\d{4}\.\d{4,5})', pdf_url)
                title = f"arXiv Paper {arxiv_match.group(1)}" if arxiv_match else "arXiv Paper"

            return ScrapedPage(
                url=pdf_url,
                title=title,
                content=full_text,
                word_count=len(full_text.split()),
                domain=domain,
                scraped_at=datetime.utcnow().isoformat(),
                success=True,
            )

        except Exception as e:
            # Ensure cleanup even on failure
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            logger.warning(f"[WebScraper] arXiv PDF extraction failed for {pdf_url}: {e}")
            return ScrapedPage(url=pdf_url, domain=domain, success=False, error=f"PDF extraction failed: {str(e)[:150]}")

    def _clean_pdf_text(self, text: str) -> str:
        """Clean text extracted from a PDF (remove headers, footers, page numbers, etc.)."""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Skip very short lines that are likely page numbers or headers
            if len(stripped) <= 3 and stripped.isdigit():
                continue
            # Skip lines that are just dashes or underscores
            if re.match(r'^[-_=]{3,}$', stripped):
                continue
            cleaned.append(stripped)

        text = "\n".join(cleaned)
        # Collapse excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()

    # ═══════════════════════════════════════════════════
    # Content Extraction (HTML)
    # ═══════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════

    def _is_blocked(self, domain: str) -> bool:
        """Check if a domain is in the blocklist."""
        domain_lower = domain.lower()
        for blocked in self.BLOCKED_DOMAINS:
            if blocked in domain_lower:
                return True
        return False

    def _is_arxiv_pdf(self, url: str) -> bool:
        """Check if a URL points to an arXiv PDF."""
        return "arxiv.org/pdf/" in url

    def _priority_score(self, url: str) -> int:
        """Score a URL for priority ordering. Higher = scraped first."""
        domain = urlparse(url).netloc.replace("www.", "").lower()
        for pd in self.PRIORITY_DOMAINS:
            if pd in domain:
                return 10
        return 0

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
