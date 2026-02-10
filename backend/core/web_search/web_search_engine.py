"""
Web Search Engine - Orchestrates multi-provider search and result fusion.
"""
import asyncio
import re
import math
from typing import List, Dict, Tuple, Optional
from collections import Counter
from loguru import logger

from backend.core.web_search.providers.base_provider import SearchResult
from backend.core.web_search.providers.duckduckgo_provider import DuckDuckGoProvider
from backend.core.web_search.providers.wikipedia_provider import WikipediaProvider
from backend.core.web_search.providers.bing_provider import BingSearchProvider
from backend.core.web_search.providers.google_scraper_provider import GoogleScraperProvider


class WebSearchEngine:
    """
    Multi-provider web search engine with intelligent result fusion.
    
    Executes searches across 4 providers in parallel, deduplicates results,
    scores them using TF-IDF relevance + source diversity + recency bonuses,
    and returns the top-K fused results for LLM synthesis.
    """
    
    def __init__(self):
        self.providers = [
            DuckDuckGoProvider(max_results=6),
            WikipediaProvider(max_results=3),
            BingSearchProvider(max_results=5),
            GoogleScraperProvider(max_results=5),
        ]
        logger.info(f"WebSearchEngine initialized with {len(self.providers)} providers")
    
    async def search(self, query: str, top_k: int = 10) -> Tuple[List[SearchResult], Dict]:
        """
        Execute multi-provider search and return fused results.
        
        Returns:
            Tuple of (fused_results, metrics_dict)
        """
        import time
        start_time = time.time()
        
        # Step 0: Clean query
        # Remove common "search for" prefixes that confuse engines
        clean_query = re.sub(
            r'^(please\s+)?(search\s+(the\s+web\s+)?for|find|google)\s+', 
            '', 
            query, 
            flags=re.IGNORECASE
        ).strip()
        
        logger.info(f"Original query: '{query}' -> Cleaned: '{clean_query}'")
        
        # Step 1: Query all providers in parallel (Pass 1)
        provider_results = await self._parallel_search(clean_query)
        
        # Step 2: Fuse results
        fused = self._fuse_results(provider_results, query, top_k)
        
        # Step 3: Deep Search (Adaptive)
        # If results are poor or query implies "latest" news, try variations
        if len(fused) < 4:
            logger.info(f"Deep Search Triggered: Initial results ({len(fused)}) insufficient. Expanding...")
            
            variations = []
            if "latest" not in clean_query.lower():
                variations.append(f"latest {clean_query}")
            if "news" not in clean_query.lower():
                variations.append(f"{clean_query} news")
            
            # Execute variation searches
            for var_query in variations[:2]: # Limit to 2 variations
                logger.info(f"Deep Search Variation: {var_query}")
                extra_results = await self._parallel_search(var_query)
                provider_results.extend(extra_results)
            
            # Re-fuse with all results
            fused = self._fuse_results(provider_results, query, top_k)
        
        elapsed = (time.time() - start_time) * 1000
        
        # Build metrics
        metrics = {
            "web_search_time_ms": round(elapsed, 1),
            "providers_queried": len(self.providers),
            "providers_with_results": sum(1 for r in provider_results if r),
            "total_raw_results": sum(len(r) for r in provider_results),
            "fused_results_count": len(fused),
            "provider_breakdown": {
                p.name: len(r) for p, r in zip(self.providers, provider_results)
            }
        }
        
        logger.info(
            f"Web search completed in {elapsed:.0f}ms: "
            f"{metrics['total_raw_results']} raw -> {len(fused)} fused results"
        )
        
        return fused, metrics
    
    async def _parallel_search(self, query: str) -> List[List[SearchResult]]:
        """Execute all provider searches in parallel."""
        tasks = [provider.search(query) for provider in self.providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions gracefully
        clean_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Provider {self.providers[i].name} raised: {result}")
                clean_results.append([])
            else:
                clean_results.append(result)
        
        return clean_results
    
    def _fuse_results(
        self, 
        all_results: List[List[SearchResult]], 
        query: str, 
        top_k: int
    ) -> List[SearchResult]:
        """
        Multi-stage result fusion algorithm.
        
        1. URL-based deduplication
        2. TF-IDF relevance scoring
        3. Source diversity bonus (multi-provider agreement)
        4. Recency bonus
        5. Top-K selection
        """
        # Step 1: Deduplicate by URL (normalized)
        seen_urls = {}
        for result_list in all_results:
            for r in result_list:
                normalized_url = self._normalize_url(r.url)
                if normalized_url not in seen_urls:
                    seen_urls[normalized_url] = r
                else:
                    # Track that multiple providers found this URL
                    seen_urls[normalized_url].provider_count += 1
        
        unique_results = list(seen_urls.values())
        
        if not unique_results:
            return []
        
        # Step 2: Compute TF-IDF relevance scores
        query_terms = self._tokenize(query.lower())
        for r in unique_results:
            text = f"{r.title} {r.snippet}".lower()
            doc_terms = self._tokenize(text)
            r.relevance_score = self._compute_tfidf(query_terms, doc_terms)
        
        # Step 3: Diversity bonus (reward multi-provider agreement)
        for r in unique_results:
            r.diversity_bonus = (r.provider_count - 1) * 0.15
        
        # Step 4: Recency bonus
        for r in unique_results:
            r.recency_score = self._compute_recency(r.date)
        
        # Step 5: Compute final score and sort
        for r in unique_results:
            r.final_score = (
                r.relevance_score * 0.6 +   # Relevance is most important
                r.diversity_bonus * 0.25 +   # Multi-provider agreement
                r.recency_score * 0.15       # Recency matters less
            )
        
        unique_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return unique_results[:top_k]
    
    def _normalize_url(self, url: str) -> str:
        """Normalize a URL for deduplication."""
        url = url.strip().rstrip("/")
        url = re.sub(r'^https?://(www\.)?', '', url)
        # Remove query parameters and fragments
        url = url.split("?")[0].split("#")[0]
        return url.lower()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for TF-IDF."""
        # Remove non-alphanumeric characters, split on whitespace
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # Remove very short tokens and common stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'and', 'but', 'or', 'not', 'no', 'nor',
            'so', 'yet', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'than', 'too', 'very', 'just', 'about', 'also',
            'it', 'its', 'this', 'that', 'these', 'those', 'what', 'which',
            'who', 'whom', 'how', 'when', 'where', 'why', 'all', 'any',
        }
        return [t for t in tokens if len(t) > 1 and t not in stopwords]
    
    def _compute_tfidf(self, query_terms: List[str], doc_terms: List[str]) -> float:
        """Compute a simplified TF-IDF similarity between query and document."""
        if not query_terms or not doc_terms:
            return 0.0
        
        doc_counter = Counter(doc_terms)
        doc_len = len(doc_terms) if doc_terms else 1
        
        score = 0.0
        for term in query_terms:
            tf = doc_counter.get(term, 0) / doc_len
            # Use log-scaled term frequency
            if tf > 0:
                score += (1 + math.log(1 + tf))
        
        # Normalize by query length
        return score / len(query_terms)
    
    def _compute_recency(self, date_str: Optional[str]) -> float:
        """Compute a recency score (0.0 to 1.0) based on date."""
        if not date_str:
            return 0.0  # Unknown date gets no bonus
        
        try:
            from datetime import datetime, timezone
            
            # Try common date formats
            for fmt in [
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
                "%B %d, %Y",
            ]:
                try:
                    dt = datetime.strptime(date_str[:19], fmt)
                    break
                except ValueError:
                    continue
            else:
                return 0.0
            
            now = datetime.now()
            days_old = (now - dt).days
            
            if days_old <= 7:
                return 1.0
            elif days_old <= 30:
                return 0.8
            elif days_old <= 90:
                return 0.6
            elif days_old <= 365:
                return 0.4
            elif days_old <= 730:
                return 0.2
            else:
                return 0.1
                
        except Exception:
            return 0.0
    
    def format_for_llm(self, results: List[SearchResult]) -> str:
        """Format fused results as context for LLM synthesis."""
        if not results:
            return "No web search results found."
        
        formatted_parts = []
        for i, r in enumerate(results, 1):
            part = f"[{i}] {r.title}\n"
            part += f"    Source: {r.url}\n"
            part += f"    Provider: {r.source_provider}\n"
            if r.snippet:
                part += f"    Content: {r.snippet}\n"
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)
