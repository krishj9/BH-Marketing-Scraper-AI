"""Pipeline orchestrator for the BH Marketing Scraper.

Ties together all pipeline stages:
1. Crawl site with Playwright (exact match during crawl)
2. If no exact match → chunk pages → embed → vector similarity
3. For borderline vector results → LLM RAG confirmation
4. Persist all results to BigQuery
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from playwright.async_api import async_playwright

from src.config import load_config, load_sites
from src.crawler.crawler import crawl_site
from src.embeddings.chunker import chunk_text
from src.embeddings.client import EmbeddingClient
from src.llm.client import LLMClient
from src.matching.vector import compute_similarities, filter_by_threshold
from src.models import ChunkInfo, CrawlSummary, SiteConfig
from src.storage.bigquery_client import BigQueryClient

logger = logging.getLogger(__name__)


async def run_pipeline(config: dict[str, Any]) -> None:
    """Run the full scraper pipeline.

    Processes sites sequentially: crawl → exact match → vector → LLM → persist.
    Each site is fully processed before moving to the next.

    Args:
        config: Application configuration dict.
    """
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    run_mode = config.get("run_mode", "local")
    keywords = config.get("keywords", [])

    logger.info("=== Pipeline starting: run_id=%s, mode=%s ===", run_id, run_mode)

    # Load target sites
    sites = load_sites(config)
    if not sites:
        logger.warning("No sites to process. Check %s", config.get("site_list_path"))
        return

    logger.info("Processing %d sites", len(sites))

    # Initialize GCP clients
    gcp = config["gcp"]
    bq = BigQueryClient(
        project_id=gcp["project_id"],
        dataset_id=gcp["bigquery_dataset"],
        location=gcp.get("region", "us-east4"),
    )
    bq.ensure_tables_exist()
    bq.insert_run_metadata(run_id, started_at, run_mode)

    vertex_config = config.get("vertex_ai", {})
    embedding_client = EmbeddingClient(
        project_id=gcp["project_id"],
        region=gcp.get("region", "us-east4"),
        model_name=vertex_config.get("embedding_model", "text-embedding-004"),
        batch_size=vertex_config.get("embedding_batch_size", 250),
    )
    llm_client = LLMClient(
        project_id=gcp["project_id"],
        region=gcp.get("region", "us-east4"),
        model_name=vertex_config.get("generative_model", "gemini-1.5-flash"),
    )

    sites_processed = 0
    sites_failed = 0

    # Launch shared Playwright browser
    async with async_playwright() as p:
        headless = config.get("crawl", {}).get("headless", True)
        browser = await p.chromium.launch(headless=headless)
        logger.info("Browser launched (headless=%s)", headless)

        for idx, site in enumerate(sites, 1):
            logger.info(
                "--- Site %d/%d: %s (%s) ---",
                idx, len(sites), site.url, site.site_id,
            )
            try:
                await _process_site(
                    site=site,
                    config=config,
                    keywords=keywords,
                    run_id=run_id,
                    browser=browser,
                    bq=bq,
                    embedding_client=embedding_client,
                    llm_client=llm_client,
                )
                sites_processed += 1
            except Exception as e:
                logger.error(
                    "Failed to process site %s: %s", site.url, e, exc_info=True
                )
                sites_failed += 1

        await browser.close()

    # Finalize run
    bq.update_run_completed(run_id, sites_processed, sites_failed)
    logger.info(
        "=== Pipeline complete: processed=%d, failed=%d, run_id=%s ===",
        sites_processed, sites_failed, run_id,
    )


async def _process_site(
    site: SiteConfig,
    config: dict[str, Any],
    keywords: list[str],
    run_id: str,
    browser: Any,
    bq: BigQueryClient,
    embedding_client: EmbeddingClient,
    llm_client: LLMClient,
) -> None:
    """Process a single site through the full pipeline.

    Args:
        site: Target site configuration.
        config: Application configuration.
        keywords: Keyword list for matching.
        run_id: Current run identifier.
        browser: Shared Playwright browser.
        bq: BigQuery client.
        embedding_client: Vertex AI embedding client.
        llm_client: Gemini LLM client.
    """
    # ── Stage 1: Crawl + Exact Match ───────────────────────────────
    crawl_result = await crawl_site(
        site_url=site.url,
        site_id=site.site_id,
        config=config,
        keywords=keywords,
        browser=browser,
    )

    # Persist crawl data to BigQuery
    _persist_crawl_data(bq, run_id, site.site_id, crawl_result)

    if crawl_result.exact_match_found:
        bq.insert_scraper_result(
            run_id=run_id,
            site_id=site.site_id,
            url=crawl_result.exact_match_url,
            match_type="exact",
            matched_keyword=crawl_result.exact_match_keyword,
        )
        logger.info(
            "Site %s: EXACT MATCH on '%s' at %s — skipping semantic search",
            site.site_id,
            crawl_result.exact_match_keyword,
            crawl_result.exact_match_url,
        )
        # Archive visited list
        if config.get("rebuild_visited_list_after_run", True):
            bq.delete_visited_pages(run_id, site.site_id)
        return

    # ── Stage 2: Vector Search ─────────────────────────────────────
    logger.info("Site %s: No exact match — running vector search", site.site_id)

    # Get pages with content (from in-memory crawl results, not BigQuery)
    pages_with_text = [
        p for p in crawl_result.pages
        if p.normalized_text and not p.blocked_reason
    ]

    if not pages_with_text:
        logger.warning("Site %s: No pages with content for vector search", site.site_id)
        if config.get("rebuild_visited_list_after_run", True):
            bq.delete_visited_pages(run_id, site.site_id)
        return

    matching_config = config.get("matching", {})
    max_chars = matching_config.get("max_chars_per_chunk", 1500)
    chunk_overlap = matching_config.get("max_chars_chunk_overlap", 100)
    emb_threshold = matching_config.get("embedding_similarity_threshold", 0.70)
    llm_threshold = matching_config.get("llm_confidence_threshold", 0.85)

    # Chunk all pages
    all_chunks: list[ChunkInfo] = []
    for page in pages_with_text:
        chunks = chunk_text(page.normalized_text, max_chars=max_chars, overlap=chunk_overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append(ChunkInfo(url=page.url, text=chunk, chunk_index=i))

    if not all_chunks:
        logger.warning("Site %s: No chunks generated", site.site_id)
        if config.get("rebuild_visited_list_after_run", True):
            bq.delete_visited_pages(run_id, site.site_id)
        return

    logger.info(
        "Site %s: %d pages → %d chunks for embedding",
        site.site_id, len(pages_with_text), len(all_chunks),
    )

    # Compute embeddings
    chunk_texts = [c.text for c in all_chunks]
    chunk_embeddings = embedding_client.get_embeddings(chunk_texts, task_type="RETRIEVAL_DOCUMENT")

    # Compute keyword embedding
    keyword_text = "; ".join(keywords)
    keyword_embedding = embedding_client.get_query_embedding(keyword_text)

    # Compute similarities
    similarities = compute_similarities(chunk_embeddings, keyword_embedding)
    above_threshold = filter_by_threshold(similarities, emb_threshold)

    logger.info(
        "Site %s: %d chunks above embedding threshold (%.2f)",
        site.site_id, len(above_threshold), emb_threshold,
    )

    # Process results: high confidence vs borderline
    result_rows = []
    for chunk_idx, sim_score in above_threshold:
        chunk_info = all_chunks[chunk_idx]

        if sim_score >= llm_threshold:
            # High confidence vector match
            result_rows.append({
                "run_id": run_id,
                "site_id": site.site_id,
                "url": chunk_info.url,
                "match_type": "vector",
                "similarity_score": sim_score,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            logger.debug(
                "Vector match (high confidence): %s score=%.3f",
                chunk_info.url, sim_score,
            )
        else:
            # Borderline: call LLM for confirmation
            logger.debug(
                "Borderline chunk (score=%.3f) for %s — calling LLM",
                sim_score, chunk_info.url,
            )
            llm_result = llm_client.analyze_content(chunk_info.url, chunk_info.text)

            if llm_result.get("has_childcare_benefit"):
                result_rows.append({
                    "run_id": run_id,
                    "site_id": site.site_id,
                    "url": chunk_info.url,
                    "match_type": "llm_confirmed",
                    "similarity_score": sim_score,
                    "llm_explanation": json.dumps(llm_result),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
                logger.info(
                    "LLM confirmed match: %s (vector=%.3f, llm_conf=%.2f)",
                    chunk_info.url, sim_score,
                    llm_result.get("confidence", 0),
                )

    # Deduplicate results by URL (keep highest similarity per URL)
    unique_results = _deduplicate_results(result_rows)

    # Persist results
    if unique_results:
        bq.insert_scraper_results_batch(unique_results)
        logger.info(
            "Site %s: %d results stored (%d vector, %d llm_confirmed)",
            site.site_id,
            len(unique_results),
            sum(1 for r in unique_results if r["match_type"] == "vector"),
            sum(1 for r in unique_results if r["match_type"] == "llm_confirmed"),
        )
    else:
        logger.info("Site %s: No matches found (exact, vector, or LLM)", site.site_id)

    # Archive visited list
    if config.get("rebuild_visited_list_after_run", True):
        bq.delete_visited_pages(run_id, site.site_id)


def _persist_crawl_data(
    bq: BigQueryClient, run_id: str, site_id: str, crawl: CrawlSummary
) -> None:
    """Batch-persist crawl data to BigQuery."""
    now = datetime.now(timezone.utc).isoformat()

    visited_rows = []
    content_rows = []

    for page in crawl.pages:
        visited_rows.append({
            "run_id": run_id,
            "site_id": site_id,
            "url": page.url,
            "http_status": page.http_status,
            "crawl_timestamp": now,
            "blocked_reason": page.blocked_reason,
        })

        if page.normalized_text and not page.blocked_reason:
            content_rows.append({
                "run_id": run_id,
                "site_id": site_id,
                "url": page.url,
                "html": page.html,
                "normalized_text": page.normalized_text,
                "created_at": now,
            })

    bq.insert_visited_pages_batch(visited_rows)
    bq.insert_page_content_batch(content_rows)
    logger.debug(
        "Persisted %d visited pages and %d content records for %s",
        len(visited_rows), len(content_rows), site_id,
    )


def _deduplicate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate results by URL, keeping highest similarity per URL."""
    best_by_url: dict[str, dict[str, Any]] = {}
    for row in rows:
        url = row["url"]
        existing = best_by_url.get(url)
        if existing is None or (row.get("similarity_score") or 0) > (existing.get("similarity_score") or 0):
            best_by_url[url] = row
    return list(best_by_url.values())
