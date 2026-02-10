"""
Search providers for the Web Search module.
"""
from backend.core.web_search.providers.duckduckgo_provider import DuckDuckGoProvider
from backend.core.web_search.providers.wikipedia_provider import WikipediaProvider
from backend.core.web_search.providers.searxng_provider import SearXNGProvider
from backend.core.web_search.providers.google_scraper_provider import GoogleScraperProvider

__all__ = [
    "DuckDuckGoProvider",
    "WikipediaProvider", 
    "SearXNGProvider",
    "GoogleScraperProvider"
]
