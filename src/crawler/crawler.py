"""Async BFS web crawler using Playwright and BeautifulSoup.

Custom BFS crawler (permitted by spec: "Custom BFS crawler acceptable
if preferred") that integrates directly with the async pipeline,
avoiding Scrapy's Twisted reactor complexity.

Features:
- BFS crawl with configurable depth and page limits
- Playwright for JS-rendered pages (SPA support)
- BeautifulSoup for text extraction and link parsing
- robots.txt compliance
- Politeness delays and retry with exponential backoff
- Exact match short-circuit during crawl
"""

import asyncio
import logging
from typing import Any, Optional
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
from playwright.async_api import Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeout

from src.matching.exact import ExactMatcher
from src.models import CrawlSummary, ExactMatchResult, PageResult

logger = logging.getLogger(__name__)

# File extensions to skip during crawl
_SKIP_EXTENSIONS = frozenset({
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".mp4", ".mp3", ".avi", ".mov", ".zip", ".tar", ".gz", ".rar",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
})


def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication.

    Removes fragments, lowercases scheme/host, strips trailing slashes.
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    normalized = urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        path,
        parsed.params,
        parsed.query,
        "",  # Remove fragment
    ))
    return normalized


def is_same_domain(url: str, base_domain: str) -> bool:
    """Check if a URL belongs to the same domain (with/without www)."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    base = base_domain.lower()
    if base.startswith("www."):
        base = base[4:]
    return domain == base


def should_skip_url(url: str) -> bool:
    """Check if URL should be skipped based on extension or scheme."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return True
    path_lower = parsed.path.lower()
    return any(path_lower.endswith(ext) for ext in _SKIP_EXTENSIONS)


def _load_robots(site_url: str, user_agent: str) -> RobotFileParser:
    """Load and parse robots.txt for the given site."""
    rp = RobotFileParser()
    robots_url = urljoin(site_url, "/robots.txt")
    rp.set_url(robots_url)
    try:
        rp.read()
        logger.debug("Loaded robots.txt from %s", robots_url)
    except Exception as e:
        logger.debug("Could not load robots.txt from %s: %s (allowing all)", robots_url, e)
    return rp


async def _fetch_page(
    context: BrowserContext,
    url: str,
    timeout_ms: int,
) -> tuple[str, int, Optional[str]]:
    """Fetch a page using Playwright and return (html, status, error).

    Args:
        context: Playwright browser context.
        url: URL to fetch.
        timeout_ms: Navigation timeout in milliseconds.

    Returns:
        Tuple of (html_content, http_status, error_message_or_none).
    """
    page: Optional[Page] = None
    try:
        page = await context.new_page()
        response = await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")

        # Allow a short time for JS rendering after DOM is loaded
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except PlaywrightTimeout:
            pass  # Use whatever content loaded

        html = await page.content()
        status = response.status if response else 0
        return html, status, None

    except PlaywrightTimeout:
        # Timeout: try to get partial content
        if page:
            try:
                html = await page.content()
                logger.warning("Timeout fetching %s, using partial content", url)
                return html, 0, None
            except Exception:
                pass
        return "", 0, f"timeout after {timeout_ms}ms"

    except Exception as e:
        return "", 0, str(e)[:300]

    finally:
        if page:
            try:
                await page.close()
            except Exception:
                pass


def _extract_text(html: str) -> str:
    """Extract normalized text from HTML using BeautifulSoup.

    Removes script, style, and noscript tags before extraction.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text


def _extract_links(html: str, base_url: str, base_domain: str) -> list[str]:
    """Extract valid same-domain links from HTML."""
    soup = BeautifulSoup(html, "lxml")
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        full_url = urljoin(base_url, href)
        if is_same_domain(full_url, base_domain) and not should_skip_url(full_url):
            links.append(normalize_url(full_url))
    return links


async def crawl_site(
    site_url: str,
    site_id: str,
    config: dict[str, Any],
    keywords: list[str],
    browser: Browser,
) -> CrawlSummary:
    """Crawl a site using async BFS with Playwright.

    Performs exact keyword matching during crawl. If any page matches,
    the crawl short-circuits and returns immediately.

    Args:
        site_url: Base URL to start crawling.
        site_id: Normalized site identifier.
        config: Crawl configuration dict.
        keywords: List of keywords for exact matching.
        browser: Shared Playwright browser instance.

    Returns:
        CrawlSummary with all pages and exact match info.
    """
    crawl_config = config["crawl"]
    max_pages = crawl_config["max_pages_per_site"]
    max_depth = crawl_config["max_depth"]
    delay = crawl_config["request_delay_seconds"]
    timeout_ms = crawl_config["page_timeout_ms"]
    max_retries = crawl_config.get("max_retries", 3)
    user_agent = crawl_config.get("user_agent", "BHMarketingScraper/1.0")

    exact_matcher = ExactMatcher(keywords)

    # Load robots.txt
    robots = _load_robots(site_url, user_agent)

    # Setup browser context with custom user agent
    context = await browser.new_context(user_agent=user_agent)

    base_domain = urlparse(site_url).netloc.lower()
    start_url = normalize_url(site_url)

    # BFS state
    queue: list[tuple[str, int]] = [(start_url, 0)]  # (url, depth)
    visited: set[str] = set()
    pages: list[PageResult] = []
    pages_blocked = 0
    exact_match: Optional[ExactMatchResult] = None

    logger.info("Starting crawl of %s (max_pages=%d, max_depth=%d)", site_url, max_pages, max_depth)

    try:
        while queue and len(pages) < max_pages:
            url, depth = queue.pop(0)

            # Skip already visited
            if url in visited:
                continue
            visited.add(url)

            # Skip if exceeds depth
            if depth > max_depth:
                continue

            # Check robots.txt
            if not robots.can_fetch(user_agent, url):
                logger.debug("Blocked by robots.txt: %s", url)
                pages_blocked += 1
                pages.append(PageResult(
                    url=url, html="", normalized_text="",
                    http_status=0, blocked_reason="robots_txt",
                ))
                continue

            # Fetch with retries
            html, status, error = "", 0, "max retries exceeded"
            for attempt in range(max_retries):
                html, status, error = await _fetch_page(context, url, timeout_ms)
                if error is None:
                    break
                if attempt < max_retries - 1:
                    backoff = (2 ** attempt) * 1.0  # 1s, 2s, 4s
                    logger.warning(
                        "Retry %d/%d for %s (error: %s), waiting %.1fs",
                        attempt + 1, max_retries, url, error, backoff,
                    )
                    await asyncio.sleep(backoff)

            if error:
                pages_blocked += 1
                pages.append(PageResult(
                    url=url, html="", normalized_text="",
                    http_status=status, blocked_reason=f"error: {error}",
                ))
                continue

            # Extract text
            text = _extract_text(html)

            page_result = PageResult(
                url=url, html=html, normalized_text=text, http_status=status,
            )
            pages.append(page_result)

            logger.debug(
                "Crawled [%d/%d] depth=%d status=%d chars=%d: %s",
                len(pages), max_pages, depth, status, len(text), url,
            )

            # Check exact match (short-circuit)
            match = exact_matcher.match(text)
            if match:
                match.url = url
                exact_match = match
                logger.info(
                    "Exact match found on %s: keyword='%s' — stopping crawl",
                    url, match.keyword,
                )
                break

            # Extract and enqueue links
            if depth < max_depth:
                links = _extract_links(html, url, base_domain)
                for link in links:
                    if link not in visited:
                        queue.append((link, depth + 1))

            # Politeness delay
            await asyncio.sleep(delay)

    finally:
        await context.close()

    pages_crawled = len([p for p in pages if not p.blocked_reason])
    summary = CrawlSummary(
        site_id=site_id,
        site_url=site_url,
        pages_crawled=pages_crawled,
        pages_blocked=pages_blocked,
        exact_match_found=exact_match is not None,
        exact_match_url=exact_match.url if exact_match else None,
        exact_match_keyword=exact_match.keyword if exact_match else None,
        pages=pages,
    )

    logger.info(
        "Crawl complete for %s: %d pages crawled, %d blocked, exact_match=%s",
        site_url, pages_crawled, pages_blocked, summary.exact_match_found,
    )
    return summary
