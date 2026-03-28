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
    - Locale-aware URL deduplication (prevents IBM-in-5-languages problem)
    - URL relevance pre-filtering (prevents stackoverflow/chanel.com scraping)
    - Blocked domain filtering (Wikipedia, StackOverflow, retail, etc.)
    - Content length limiting
    - Timeout handling
    - Domain extraction
    """

    # ─── Blocked Domains ─────────────────────────────────
    # These are skipped entirely — social media, wikis, low-quality, and
    # off-topic sites that cause false positives for research queries
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
        "scribd.com",       # paywall
        "slideshare.net",
        "reddit.com",       # noisy forum content

        # Programming Q&A — massive false positive source for research terms
        # "object", "instance", "channel", "detection" are all programming terms
        "stackoverflow.com",
        "stackexchange.com",
        "superuser.com",
        "serverfault.com",
        "askubuntu.com",
        "mathoverflow.net",
        "crossvalidated.com",

        # Programming language references — false positive for "future", "iterator", etc.
        "cppreference.com",
        "cplusplus.com",
        "docs.python.org",
        "developer.mozilla.org",
        "docs.oracle.com",
        "learn.microsoft.com",
        "docs.microsoft.com",

        # Generic tech support (not research content)
        "support.google.com",
        "support.microsoft.com",
        "support.apple.com",
        "help.github.com",

        # E-commerce / retail — false positive for "channel", "detection", etc.
        "chanel.com",
        "amazon.com",
        "ebay.com",
        "etsy.com",
        "walmart.com",
        "shopify.com",

        # Chinese Q&A / non-English platforms (irrelevant for English research)
        "zhihu.com",
        "baidu.com",
        "weibo.com",
        "csdn.net",

        # Japanese Q&A portals — massive source of hallucination-inducing noise
        # (search engines return these for Japanese-language queries on any topic)
        "chiebukuro.yahoo.co.jp",
        "detail.chiebukuro.yahoo.co.jp",
        "yahoo.co.jp",          # entire Japanese Yahoo network
        "oshiete.goo.ne.jp",    # Japanese Q&A (goo answers)
        "okwave.jp",            # Japanese Q&A platform
        "oshiete1.goo.ne.jp",

        # Korean portals / wikis
        "namu.wiki",
        "naver.com",
        "kin.naver.com",

        # Japanese / other non-English forums
        "hinative.com",
        "lang-8.com",
        "tc-forum.co.jp",
        "chiphell.com",

        # General news sites (not research sources)
        "ndtv.com",
        "timesofindia.com",
        "hindustantimes.com",
        "indiatoday.in",
        "thehindu.com",
        "news18.com",
        "indianexpress.com",

        # Generic tech forums (low research signal)
        "tomsguide.com",
        "tomshardware.com",
        "forums.macrumors.com",
        "macrumors.com",
        "techradar.com",
        "cnet.com",
        "zdnet.com",
        "theverge.com",
        "engadget.com",
        "wired.com",
        "gizmodo.com",

        # Corporate / pharma / healthcare / consumer (false positives for name collisions)
        "jnj.com", "investor.jnj.com", "careers.jnj.com",
        "pfizer.com", "merck.com", "bayer.com",

        # Gaming / chess / entertainment (false positives for algorithm terms)
        "lichess.org", "en.lichess.org", "lichess.my",
        "chess.com", "play.google.com", "apps.apple.com",
        "filmweb.pl", "imdb.com",
        "wwe.com", "espn.com", "espncricinfo.com",
        "transfermarkt.com", "transfermarkt.co.in", "fcbayern.com",
        "nzc.nz",

        # Non-English forums and portals (irrelevant for English research)
        "forum.benchmark.pl", "benchmark.pl",
        "dxy.cn",  # Chinese medical Q&A
        "iciba.com",  # Chinese dictionary
        "polimetro.com", "cohotech.vn", "mundowin.com",
        "vi.windowsnoticias.com", "vi.101-help.com",
        "pchardwarepro.com", "tinhn.com", "smartcr.org",
        "filmweb.pl",
        "pt.stackoverflow.com", "es.stackoverflow.com",
        "spanish.stackexchange.com",

        # Government data portals (false positives for generic terms)
        "data.gov.in", "data.gov",
        "iras.gov.sg", "eservices.mas.gov.sg", "customs.gov.sg",
        "ncert.nic.in",

        # Generic learning portals (too broad for deep research)
        "w3schools.com",
        "simplilearn.com",
        "numberanalytics.com",

        # Wolfram product / marketing / function-reference pages
        # These appear for any query containing "mathematical", "function", etc.
        # but contain no research-specific content (no model papers, benchmarks, etc.)
        "wolfram.com",
        "functions.wolfram.com",
        "reference.wolfram.com",
        "mathematica.stackexchange.com",
        # mathworld.wolfram.com intentionally kept — has legitimate math definitions

        # General answer / how-to aggregators
        "howtogeek.com",
        "makeuseof.com",
        "digitaltrends.com",

        # Yahoo Answers global (shut down but still indexed)
        "answers.yahoo.com",
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
        "openai.com", "anthropic.com",
        "proceedings.neurips.cc", "proceedings.mlr.press",
        "cv-foundation.org", "ecva.net",
    }

    # ─── Research URL Hint Keywords ──────────────────────
    # Used by _filter_by_url_relevance to prioritize research-relevant URLs
    RESEARCH_URL_HINTS = [
        "arxiv", "paper", "research", "journal", "conference",
        "ieee", "acm", "springer", "nature", "github",
        "huggingface", "openreview", "proceedings", "preprint",
        "abstract", "pdf", "publication", "scholar",
        "vision", "detection", "segmentation", "neural", "deep",
        "learning", "model", "dataset", "benchmark", "survey",
        "transformer", "convolution", "attention",
    ]

    # ─── Locale Pattern for URL Deduplication ────────────
    # Matches patterns like /en-us/, /ja-jp/, /de-de/, /zh-cn/ etc.
    LOCALE_PATTERN = re.compile(
        r'/(en|ja|de|fr|pt|ko|zh|es|it|nl|ru|ar|tr|pl|sv|da|fi|nb|cs|hu|ro|uk)'
        r'[-_]'
        r'(us|gb|jp|de|fr|br|kr|cn|es|it|nl|ru|ar|tr|pl|se|dk|fi|no|cz|hu|ro|ua|in|au|ca|mx)'
        r'(?=/|$)',
        re.IGNORECASE
    )

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

        Pipeline:
        1. Normalize and deduplicate URLs by exact URL
        2. Filter blocked domains
        3. Deduplicate by locale-normalized path (prevents multi-language duplicates)
        4. Pre-filter by URL relevance (prioritizes research-relevant URLs)
        5. Sort by priority domain score
        6. Scrape in parallel with semaphore

        Args:
            urls: List of URLs to scrape

        Returns:
            List of ScrapedPage results (one per URL)
        """
        # Step 1: Normalize and exact-deduplicate
        seen = set()
        unique_urls = []
        for url in urls:
            normalized = self._normalize_url(url)
            if normalized and normalized not in seen:
                seen.add(normalized)
                domain = urlparse(normalized).netloc.replace("www.", "")
                # Step 2: Check if domain is blocked
                if self._is_blocked(domain):
                    logger.info(f"[WebScraper] Blocked domain skipped: {domain}")
                    continue
                unique_urls.append(normalized)

        if not unique_urls:
            return []

        # Step 3: Deduplicate by locale-normalized path
        # Prevents scraping IBM XAI page in English, Japanese, German, etc.
        unique_urls = self._deduplicate_by_content_path(unique_urls)

        # Step 4: Pre-filter by URL relevance
        # Moves clearly irrelevant URLs (stackoverflow, chanel.com) to end of list
        unique_urls = self._filter_by_url_relevance(unique_urls)

        # Step 5: Sort — priority domains first, then relevance-ordered
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

    # ═══════════════════════════════════════════════════
    # URL Pre-Processing Methods
    # ═══════════════════════════════════════════════════

    def _deduplicate_by_content_path(self, urls: List[str]) -> List[str]:
        """
        Remove URLs that are likely translations/localizations of the same page.

        Detects patterns like:
            ibm.com/en-us/topics/xai   <- keep (English preferred)
            ibm.com/ja-jp/topics/xai   <- remove (same content, Japanese)
            ibm.com/de-de/topics/xai   <- remove (same content, German)

        Strategy: strip locale segments and deduplicate on (domain, canonical_path).
        When multiple locale variants exist, the English version is preferred.
        """
        seen_canonical = {}  # canonical_key -> url
        result = []

        for url in urls:
            try:
                parsed = urlparse(url)
                # Strip locale segment from path to get canonical path
                canonical_path = self.LOCALE_PATTERN.sub('', parsed.path)
                # Also normalize trailing slashes and query params
                canonical_path = canonical_path.rstrip('/')
                canonical_key = f"{parsed.netloc.lower()}{canonical_path.lower()}"

                if canonical_key not in seen_canonical:
                    seen_canonical[canonical_key] = url
                    result.append(url)
                else:
                    # If we already have this page but current URL is English, prefer it
                    existing_url = seen_canonical[canonical_key]
                    url_lower = url.lower()
                    existing_lower = existing_url.lower()
                    is_english = ('/en-us/' in url_lower or '/en-gb/' in url_lower
                                  or '/en/' in url_lower or '/english/' in url_lower)
                    existing_is_english = ('/en-us/' in existing_lower or '/en-gb/' in existing_lower
                                           or '/en/' in existing_lower or '/english/' in existing_lower)
                    if is_english and not existing_is_english:
                        # Replace with English version
                        seen_canonical[canonical_key] = url
                        result = [u if u != existing_url else url for u in result]
                        logger.debug(f"[WebScraper] Preferred English locale: {url} over {existing_url}")
                    else:
                        logger.debug(f"[WebScraper] Locale duplicate skipped: {url} (canonical: {canonical_key})")

            except Exception:
                # If parsing fails, include the URL to be safe
                result.append(url)

        original_count = len(urls)
        deduped_count = len(result)
        if original_count != deduped_count:
            logger.info(f"[WebScraper] Locale dedup: {original_count} -> {deduped_count} URLs")

        return result

    def _filter_by_url_relevance(self, urls: List[str]) -> List[str]:
        """
        Reorder URLs so research-relevant ones come first.

        Checks if any research hint keyword appears in the URL's domain or path.
        URLs without any hint are moved to the end of the list (not removed,
        since they might still contain relevant content).

        This prevents wasting the per-step scrape budget on clearly off-topic
        pages like stackoverflow.com/questions/what-is-object or chanel.com.
        """
        high_confidence = []
        low_confidence = []

        for url in urls:
            url_lower = url.lower()
            if any(hint in url_lower for hint in self.RESEARCH_URL_HINTS):
                high_confidence.append(url)
            else:
                low_confidence.append(url)
                logger.debug(f"[WebScraper] Low-confidence URL deprioritized: {url}")

        if low_confidence:
            logger.info(
                f"[WebScraper] URL relevance filter: {len(high_confidence)} high-confidence, "
                f"{len(low_confidence)} low-confidence (moved to end)"
            )

        return high_confidence + low_confidence

    # ═══════════════════════════════════════════════════
    # Core Scraping
    # ═══════════════════════════════════════════════════

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
                main_content = (
                    soup.find("article")
                    or soup.find("main")
                    or soup.find("div", class_=re.compile(r"content|article|post"))
                )

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
        # Give a small boost to URLs that contain research hint keywords
        url_lower = url.lower()
        if any(hint in url_lower for hint in self.RESEARCH_URL_HINTS):
            return 5
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