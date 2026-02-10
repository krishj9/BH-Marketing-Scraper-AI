"""Shared data models for the BH Marketing Scraper pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse


@dataclass
class SiteConfig:
    """A target site to crawl."""

    url: str
    name: Optional[str] = None

    @property
    def site_id(self) -> str:
        """Normalized site identifier from URL (domain without www.)."""
        parsed = urlparse(self.url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain


@dataclass
class PageResult:
    """Result from crawling a single page."""

    url: str
    html: str
    normalized_text: str
    http_status: int
    blocked_reason: Optional[str] = None


@dataclass
class ExactMatchResult:
    """Result from exact keyword matching."""

    matched: bool
    keyword: Optional[str] = None
    url: Optional[str] = None


@dataclass
class ChunkInfo:
    """A text chunk from a page, used for embedding."""

    url: str
    text: str
    chunk_index: int = 0


@dataclass
class VectorMatchResult:
    """Result from vector similarity search."""

    url: str
    chunk_text: str
    similarity: float
    match_type: str = "vector"  # vector | llm_confirmed
    matched_keyword: Optional[str] = None
    llm_explanation: Optional[str] = None


@dataclass
class CrawlSummary:
    """Summary of a site crawl including all page data."""

    site_id: str
    site_url: str
    pages_crawled: int
    pages_blocked: int
    exact_match_found: bool
    exact_match_url: Optional[str] = None
    exact_match_keyword: Optional[str] = None
    pages: list[PageResult] = field(default_factory=list)


@dataclass
class RunMetadata:
    """Metadata for a pipeline run."""

    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    sites_processed: int = 0
    sites_failed: int = 0
    run_mode: str = "local"
