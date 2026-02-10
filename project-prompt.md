# Hybrid Scraper Using Scrapy, Playwright, and BeautifulSoup

## Overview

This document is the single source of truth for implementing a **manual-triggered pipeline** that crawls 50–100 potential client sites, performs site-wide exact keyword matching, and only if no exact match is found anywhere on a site, performs a site-wide semantic search using vector embeddings with optional LLM RAG for borderline cases and explanations. Persist a complete visited-page list per site and rebuild it for each new run.

## Objective and High-Level Flow

### Objective

Crawl a provided list of potential client websites, run exact keyword matching across every visited page on each site, and if no exact match exists anywhere on the site, run a site-wide semantic search using embeddings. Use LLM RAG only for borderline vector results or to generate explanations.

### High-Level Flow

1. **Manual trigger** starts a site run.
2. **Full site crawl** discovers and persists all visited pages for the site.
3. **Stage 1 – Exact Matching** runs on every visited page during crawl. If any page meets the exact match threshold, record results and **skip Stage 2 for that site**.
4. **Decision Gate**: exact match anywhere → skip semantic. No exact match anywhere → Stage 2.
5. **Stage 2 – Vector Search** computes embeddings for all persisted pages and runs vector similarity against keyword embeddings.
6. **LLM RAG** invoked only for borderline vector results to confirm and explain.
7. Store final results, archive or delete the visited-page list, and rebuild fresh for the next manual run.

## Design Explanation (Easy to Understand)

### The Problem We're Solving

Right now, the system only finds exact keyword matches. If a company website says "childcare subsidy," we find it. But if they say "family care financial assistance" or "employee dependent care support," we miss it—even though it means the same thing. This is like having a search that only finds exact word matches, not understanding what words mean.

### The Solution: Site-Level Two-Stage Detection

**Step 1: Exact Keyword Check During Crawling**

- As we crawl each page, we quickly scan for known keywords (e.g., "childcare," "daycare," "subsidy")
- **Simple exact match**—no scoring, no weights. If a keyword appears, it's a match.
- If **any page** on the site has an exact match → record it and **skip Stage 2 for the entire site**.

**Step 2: Site-Wide Vector Search (Only If No Exact Match)**

- **Only if no exact match exists anywhere on the site**, we run semantic search.
- Compute embeddings for all persisted page chunks using Vertex AI `gemini-embedding-001`.
- Compare page embeddings to keyword embeddings via cosine similarity.
- High-similarity results → match.
- **Borderline** (similarity between thresholds) → call LLM RAG to confirm and explain.

### Why This Approach?

- **Site-level short-circuit**: One exact match anywhere avoids expensive semantic search for the whole site.
- **Simple keyword logic**: No scoring or weighting—just exact substring matching.
- **Efficient**: Vector embeddings are primary for semantic search; LLM only for borderline cases.
- **Persist and rebuild**: Full visited-page list per site, rebuilt fresh each run.

## GCP Configuration

| Setting | Value |
|---------|-------|
| **Project ID** | `mybrightday-dev` |
| **Region** | `us-east4` |
| **BigQuery Dataset** | `bh_scraper` (location: us-east4) |
| **Cloud Run Job** | `bh-scraper-job` (deploy to us-east4) |
| **Vertex AI** | Use us-east4 endpoint for embeddings and Gemini |

**Secret Manager secrets** (create as needed):
- `captcha-solver-api-key` – API key for paid CAPTCHA solver (when `captcha_policy: solve_paid_service`)
- Service account keys stored in Secret Manager for local dev (optional)

## Project Structure

```
BH-Marketing-Scraper-AI/
├── src/
│   ├── main.py              # Entry point, orchestration
│   ├── crawler/             # Scrapy spiders, Playwright integration
│   ├── matching/            # Exact match detector, vector similarity
│   ├── embeddings/          # Vertex AI embedding client, chunking
│   ├── llm/                 # Gemini RAG client, prompt template
│   └── storage/             # BigQuery client, schema operations
├── config/
│   ├── config.yaml          # Default config
│   └── sites.yaml           # Manual site list (or sites.json)
├── deployment/
│   ├── Dockerfile
│   ├── cloudbuild.yaml
│   └── cloud-run-job.yaml
├── requirements.txt
└── README.md
```

## Data Flow: Page Content Caching vs Revisit

**Pages are not revisited during vector matching or LLM calls.** Content is cached locally and reused.

| Phase | Content source | Behavior |
|-------|----------------|----------|
| **Crawl** | Live fetch (Scrapy/Playwright) | Fetch HTML, extract text, **save to BigQuery** |
| **Storage** | **BigQuery** | Store HTML, normalized text, URL, metadata per page |
| **Stage 1 (Exact match)** | Uses content from BigQuery | No re-fetch |
| **Stage 2 (Vector search)** | Uses normalized text from BigQuery | No re-fetch |
| **LLM RAG** | Uses content chunks from BigQuery | No re-fetch |

**Flow per site:**

1. Crawl site → persist HTML + text to BigQuery.
2. Run exact match on stored content. If a page is blocked by robots.txt or captcha, specify in the output.
3. If no exact match → load content from BigQuery → compute embeddings → run vector similarity (in-memory) → optionally call LLM on cached chunks.
4. Store results, archive/delete visited list.
5. Move to next site only after processing (including vector/LLM) is complete for the current site.

**Implication:** BigQuery holds all page content until Stage 2 and LLM processing is done for that site. No re-crawling of the same URLs within a run.

## BigQuery Schema

Dataset: `bh_scraper` (project: `mybrightday-dev`, location: `us-east4`)

| Table | Purpose |
|-------|---------|
| `visited_pages` | URL, site_id, crawl metadata per run |
| `page_content` | HTML, normalized text, per page |
| `scraper_results` | Final matches (exact + vector + LLM) per run |
| `run_metadata` | Run-level status and counts |

**visited_pages**

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | STRING | Unique run identifier |
| `site_id` | STRING | Normalized site domain |
| `url` | STRING | Page URL |
| `http_status` | INT64 | Response status code |
| `crawl_timestamp` | TIMESTAMP | When page was crawled |
| `blocked_reason` | STRING | Optional: `robots_txt`, `captcha`, etc. |

**page_content**

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | STRING | Links to run |
| `site_id` | STRING | Site domain |
| `url` | STRING | Page URL |
| `html` | STRING | Raw HTML (optional, for audit) |
| `normalized_text` | STRING | Extracted text for matching |
| `created_at` | TIMESTAMP | Insert timestamp |

**scraper_results**

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | STRING | Run identifier |
| `site_id` | STRING | Site domain |
| `url` | STRING | Matched page URL |
| `match_type` | STRING | `exact`, `vector`, `llm_confirmed` |
| `matched_keyword` | STRING | For exact: keyword; else NULL |
| `similarity_score` | FLOAT64 | For vector/LLM: score |
| `llm_explanation` | STRING | For LLM: explanation JSON |
| `created_at` | TIMESTAMP | Insert timestamp |

**run_metadata**

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | STRING | Unique run identifier |
| `started_at` | TIMESTAMP | Run start |
| `completed_at` | TIMESTAMP | Run end (NULL if interrupted) |
| `sites_processed` | INT64 | Successful sites |
| `sites_failed` | INT64 | Failed sites |
| `run_mode` | STRING | `local` or `cloud_run` |

## Vector Similarity: Phase 1 = In-Memory Only

**Phase 1:** No vector database. Use **in-memory** cosine similarity only. Choose one for future phases:

| Option | Use case | Notes |
|--------|----------|-------|
| **In-memory** | Single-site runs, ~50–500 pages per site | Compute cosine similarity in Python (NumPy, etc.). No external DB. Embeddings held in memory for the current site only. Simple, no extra infra. |
| **Open-source vector DB** | Multi-site runs, larger scale, or persistent index | Chroma, Qdrant, LanceDB, etc. Store embeddings, run similarity search. Self-hosted or local. |
| **Vertex Matching Engine** | GCP-native, production scale | Managed vector search in GCP. Use when you need high scale or managed indexing. |

**Recommendation:** For MVP with 50–100 sites and ~50–200 pages per site, **in-memory** is sufficient: compute embeddings for the current site’s pages, store in memory, run cosine similarity against keyword embeddings, then discard before moving to the next site. No vector DB in Phase 1.


## Recommended Stack and Components

### Crawling and Discovery

- **Scrapy** for site discovery, link extraction, politeness, retries, and concurrency.
- Custom BFS crawler acceptable if preferred.

### JS Rendering and Interaction

- **Playwright** for SPA rendering, complex interactions, and CAPTCHA flows.
- Selenium ChromeDriver only as fallback.

### Parsing and Normalization

- **BeautifulSoup** for HTML parsing and normalized text extraction from snapshots.

### Semantic Stack

- **Vertex AI embeddings**: `gemini-embedding-001` (or `text-embedding-004`) for page and keyword embeddings. Use `us-east4` endpoint.
- **Vector similarity**: In-memory only in Phase 1 (no vector DB).
- **Vertex AI Generative Models**: `gemini-1.5-flash` or `gemini-1.5-pro` for RAG/explanations when needed. Use `us-east4` endpoint.

### Storage and Orchestration

- **BigQuery** for HTML, normalized text, visited-page metadata, and results.
- **Cloud Run Jobs** for cloud execution (manual trigger).
- **Local execution** supported for development and testing.
- **Secret Manager** for credentials and solver keys.
- **Cloud Logging and Monitoring** for observability.

## Keywords for Stage 1 (Exact Match) and Stage 2 (Vector Search)

**Same keyword set** is used for both Stage 1 (exact substring matching) and Stage 2 (vector embeddings and similarity search). Store these in a keywords.yaml file in config folder:

```
corporate daycare; corporate day care; onsite daycare for employees; on site child care benefits; on site daycare benefits; onsite childcare; on site childcare; on site childcare for employees; company daycare on site; onsite daycare; childcare center; child care center; employer sponsored childcare; early education center; childcare benefit; family benefits; dependent care; childcare at work; childcare for employees; childcare near my workplace; childcare near work; childcare program; company daycare; corporate child care; corporate childcare; daycare at work; daycare benefit; daycare center; daycare in office; daycare in workplace; daycare near office; employee childcare; employer childcare; employer daycare; employer provided daycare; near-site childcare; office daycare; on site care; workplace daycare; traditional child care; family support; family support benefits; caregiver support; caregiver benefits; caregiving support; childcare assistance; support for caregivers; support for parents
```

## LLM Prompt Template for Borderline Cases

**SYSTEM**

You are an expert at identifying employer provided childcare benefits in web content.

**USER**

Analyze the following page content and determine whether it describes employer provided childcare support including on site childcare, childcare subsidies or stipends, backup childcare, dependent care benefits, partnerships, or other employer childcare programs. Return a JSON object:

```json
{
  "has_childcare_benefit": bool,
  "confidence": float,
  "detected_phrases": [{"text": str, "type": str, "confidence": float}],
  "explanation": str
}
```

**INPUT**

- URL: `{url}`
- CONTENT: `{content_chunk}`

**GUIDANCE**

- Use the provided keyword concepts to guide detection.
- Ignore job postings for childcare workers unless they describe employee benefits.
- Focus on benefits, perks, compensation sections, press releases, and partnership announcements.
- Classify detected phrases as one of: `on_site`, `subsidy`, `backup`, `dependent_care`, `partnership`, `other`.

## Config Keys and Controls

### Required Config Keys

| Key | Purpose | Default |
|-----|---------|---------|
| `embedding_similarity_threshold` | Minimum similarity for vector matches | `0.70` |
| `llm_confidence_threshold` | Similarity above this = high confidence (no LLM). Between this and `embedding_similarity_threshold` = borderline (call LLM) | `0.85` |
| `max_chars_per_chunk` | Max chars per chunk for embeddings and LLM inputs | `1500` |
| `max_chars_chunk_overlap` | Overlap between consecutive chunks (for context continuity) | `100` |
| `max_pages_per_site` | Max pages to crawl per site | `200` |
| `max_depth` | Max crawl depth (link hops from seed) | `4` |
| `captcha_policy` | `skip` \| `solve_paid_service` | `skip` |
| `run_mode` | `local` \| `cloud_run` – execution environment | `local` |
| `rebuild_visited_list_after_run` | Rebuild visited list fresh each run | `true` |
| `site_list_path` | Path to site list file (YAML or JSON) | `config/sites.yaml` |

**Note:** Exact match is binary—no threshold. If any keyword substring appears in page text, it's a match.

## Site List Format

Site list is loaded from `config/sites.yaml` (or path in `site_list_path`). Format:

```yaml
# config/sites.yaml
sites:
  - url: https://example.com
    name: Example Corp
  - url: https://another-site.com
    name: Another Company
```

Or JSON:

```json
{
  "sites": [
    { "url": "https://example.com", "name": "Example Corp" },
    { "url": "https://another-site.com", "name": "Another Company" }
  ]
}
```

- `url` (required): Base URL to crawl. Must include scheme (https://).
- `name` (optional): Human-readable label for logging and results.

## Orchestrator Pseudocode

```python
manual_site_list = load_sites(config.site_list_path)

for site in manual_site_list:
    try:
        visited_pages = scrapy_discover(site.url, max_depth=config.max_depth, max_pages=config.max_pages_per_site)
        persist_visited_list(site, visited_pages)

        exact_found = False
        for page in visited_pages:
            html, blocked = snapshot(page.url)  # Playwright if dynamic else requests
            if blocked:
                save_visited_blocked(site, page.url, blocked)
                continue
            text = BeautifulSoup(html).get_text(separator=" ", strip=True)
            save_to_bigquery(site, page.url, html, text)
            match = exact_match_detector(text, keywords)  # Returns {matched: bool, keyword: str} or None
            if match and match.matched:
                record_site_match(site, page.url, match.keyword, match_type="exact")
                exact_found = True
                break

        if exact_found:
            archive_visited_list(site)
            continue

        # Stage 2: Use cached content from BigQuery (no re-fetch)
        page_chunks = chunk_pages_for_embeddings(visited_pages, max_chars=config.max_chars_per_chunk)
        page_embeddings = batch_vertex_embeddings([c.text for c in page_chunks])
        keyword_embedding = vertex_embeddings(keywords_combined_text)

        vector_results = []
        for chunk, emb in zip(page_chunks, page_embeddings):
            sim = cosine_similarity(emb, keyword_embedding)
            if sim >= config.embedding_similarity_threshold:
                vector_results.append((chunk, sim))

        final_matches = []
        for chunk, sim in vector_results:
            if sim >= config.llm_confidence_threshold:
                # High confidence: trust vector, no LLM
                final_matches.append((chunk.url, {"confidence": sim, "source": "vector"}))
            else:
                # Borderline: call LLM to confirm
                llm_result = call_gemini_llm(prompt_template, url=chunk.url, content_chunk=chunk.text)
                if llm_result.has_childcare_benefit:
                    final_matches.append((chunk.url, {"confidence": llm_result.confidence, "source": "llm_confirmed", "explanation": llm_result}))

        store_site_results(site, final_matches)
        archive_visited_list(site)

    except SiteError as e:
        log_site_failure(site, e)
        continue  # Proceed to next site
```

## Deployment: Local and Cloud Run

**Both execution modes supported:**

| Mode | Use case | Trigger |
|------|----------|---------|
| **Local** | Development, testing, ad-hoc runs | `python -m src.main` or similar CLI |
| **Cloud Run Job** | Production, manual batch runs | Manual trigger via Cloud Console, gcloud, or Cloud Scheduler (one-off) |

**Implementation notes:**

- Same codebase runs locally and on Cloud Run.
- Use environment variable or config (`run_mode: "local" | "cloud_run"`) to switch behavior (e.g., BigQuery dataset, auth).
- Local: ADC or service account key for BigQuery/Vertex AI.
- Cloud Run: Container with Playwright; use service account with BigQuery + Vertex AI permissions.
- **Manual trigger only**—no cron or automatic scheduling in Phase 1.

### Deployment Configuration

**Environment variables** (Cloud Run Job):

| Variable | Purpose |
|----------|---------|
| `GCP_PROJECT_ID` | `mybrightday-dev` |
| `GCP_REGION` | `us-east4` |
| `RUN_MODE` | `cloud_run` |
| `BIGQUERY_DATASET` | `bh_scraper` |

**Dockerfile** requirements:

- Base: Python 3.11+ (e.g. `python:3.11-slim`)
- Install Playwright browsers: `playwright install chromium` (and dependencies)
- Install application deps from `requirements.txt`
- Copy `src/`, `config/`
- CMD: `python -m src.main`

**Cloud Run Job** (`deployment/cloud-run-job.yaml`):

- Job name: `bh-scraper-job`
- Region: `us-east4`
- Service account: Must have `roles/bigquery.dataEditor`, `roles/aiplatform.user`, `roles/secretmanager.secretAccessor`
- CPU: 2, Memory: 4Gi (Playwright needs headless Chromium)
- Timeout: 2h (adjust for site count)

## Output Schema

Final results are written to BigQuery `scraper_results` (see BigQuery Schema) and can be exported. Each row:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | STRING | Run identifier |
| `site_id` | STRING | Normalized domain |
| `url` | STRING | Matched page URL |
| `match_type` | STRING | `exact`, `vector`, `llm_confirmed` |
| `matched_keyword` | STRING | For exact: keyword; else NULL |
| `similarity_score` | FLOAT64 | For vector/LLM: 0–1 |
| `llm_explanation` | STRING | For LLM: JSON with explanation, detected_phrases |

## Error Handling and Robustness

- **Site-level failures**: On crawl failure, network error, or unexpected exception for a site, log `SiteError`, skip to next site. Do not abort entire run.
- **Page-level retries**: Retry failed page fetches up to 3 times with exponential backoff (1s, 2s, 4s).
- **Blocked pages**: If robots.txt disallows or CAPTCHA blocks, record `blocked_reason` in `visited_pages`, skip page, continue crawl.
- **Vertex AI / LLM rate limits**: Implement retry with backoff for 429 responses.
- **Partial results**: Per-site results are committed to BigQuery as each site completes. Run-level metadata (started_at, completed_at, sites_processed, sites_failed) recorded in run table.

## Operational and Compliance Notes

- **Manual runs only.** No scheduler required.
- **Persist full visited-page list per site** including URL, snapshot path, normalized text, HTTP status, and crawl timestamp. Rebuild fresh for each new run.
- **Respect robots.txt and site ToS.** Only scrape publicly accessible content.
- **CAPTCHA policy**: Either skip or solve via paid solver integrated into Playwright. Log all solved CAPTCHAs for compliance.
- **Politeness**: Implement per-site rate limits, retries, and exponential backoff. Use proxy rotation only when compliant.
- **Observability**: Log pages visited, exact matches, vector queries, LLM calls, token usage, and any CAPTCHAs encountered.

## Deliverables for MVP

1. **Scrapy crawler** that discovers and persists a complete visited-page list per site.
2. **Playwright snapshots** for dynamic pages and BeautifulSoup parsing for normalized text.
3. **Exact match detector** that short-circuits semantic stage on any site-wide exact match.
4. **Site-wide vector pipeline** using Vertex AI embeddings and in-memory cosine similarity (no vector DB in Phase 1).
5. **Optional LLM RAG integration** for borderline cases using the provided prompt.
6. **BigQuery schema** for HTML, normalized text, visited-page metadata, and final results.
7. **Runbook** describing manual trigger steps, adding sites, updating keywords, and CAPTCHA handling.

## Design Notes

### Keep It Simple

- **Remove scoring logic** for keyword search. Use simple exact substring matching.
- **Site-level decision gate**: One exact match anywhere on the site → skip entire semantic stage.

### Handling Long Pages

- Chunk pages using `max_chars_per_chunk` for embeddings and LLM inputs.
- Use `max_chars_chunk_overlap` for overlap between consecutive chunks to preserve context.
- Prioritize benefit-related sections (headings, divs with benefit keywords) when available.
- Fallback to smart truncation at sentence boundaries for unstructured content.
- Chunking: split on sentence boundaries first; if single sentence exceeds `max_chars_per_chunk`, split on clause boundaries.

### Content Extraction Strategy

1. Extract sections with benefit-related keywords in headings/classes.
2. Combine key sections up to `max_chars_per_chunk`.
3. Fallback: take beginning + end of page for intro and summary.

## References

- Vertex AI Gemini API: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini
- Gemini Embedding Models: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api
- Scrapy: https://scrapy.org/
- Playwright: https://playwright.dev/python/
- Crawl4AI Documentation: https://docs.crawl4ai.com/

## Crawl4AI Evaluation

### What is Crawl4AI?

Crawl4AI is an open-source Python library for LLM-friendly web scraping and crawling. It provides:
- Asynchronous web crawler (`AsyncWebCrawler`)
- Automatic HTML-to-Markdown conversion
- Built-in LLM extraction strategies (provider-agnostic via LiteLLM)
- Support for dynamic JavaScript-rendered pages
- Adaptive crawling

### Recommendation

**For this hybrid scraper project**: Crawl4AI is **not recommended** because:
- **Preferred stack**: Scrapy + Playwright + BeautifulSoup is specified.
- **Site-level short-circuit**: Our design requires exact match first, then site-wide vector search. Crawl4AI does not natively support this flow.
- **Batch processing**: Vector embeddings and LLM RAG run after full crawl; Crawl4AI is oriented toward per-page extraction.

**Use Crawl4AI for**: Future projects starting from scratch or rapid prototyping where per-page LLM extraction is needed.

---

**Last Updated**: 2026-02-10  
**Status**: Single Source of Truth – Project Prompt for Hybrid Scraper Pipeline
