"""Web search: DuckDuckGo by default (no API key), provider-abstracted.

Returns a list of {title, url, snippet} dicts. Optionally fetches and strips the
top result pages for fuller context (off by default). This module is the only
component that reaches the internet.
"""
from __future__ import annotations

import time
from typing import Dict, List

import requests

from .config import WebSearchConfig

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _search_duckduckgo(query: str, max_results: int) -> List[Dict]:
    """DuckDuckGo text search. Handles both the new `ddgs` package and the older
    `duckduckgo_search` name."""
    try:
        from ddgs import DDGS
    except ImportError:  # pragma: no cover - fallback for older installs
        from duckduckgo_search import DDGS

    results: List[Dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href") or r.get("url", ""),
                "snippet": r.get("body") or r.get("snippet", ""),
            })
    return results


def fetch_page(url: str, char_limit: int, timeout: int) -> str:
    """Fetch a URL and return cleaned visible text, truncated to char_limit."""
    from bs4 import BeautifulSoup

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return text[:char_limit]


def search(cfg: WebSearchConfig, query: str) -> List[Dict]:
    """Run a web search per config, optionally enriching results with page text."""
    if cfg.provider == "duckduckgo":
        results = _search_duckduckgo(query, cfg.result_count)
    else:
        # Provider seam: Tavily etc. can be added here. Fall back to DDG for now.
        results = _search_duckduckgo(query, cfg.result_count)

    if cfg.fetch_pages and results:
        deadline = time.monotonic() + cfg.total_time_budget_s
        for r in results:
            if time.monotonic() >= deadline or not r["url"]:
                break
            try:
                page = fetch_page(r["url"], cfg.fetch_char_limit, cfg.request_timeout_s)
                if page:
                    r["snippet"] = page
            except Exception:
                # Keep the search snippet if the page fetch fails.
                continue
    return results
