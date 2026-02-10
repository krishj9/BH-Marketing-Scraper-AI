# BH Marketing Scraper AI

Hybrid keyword + semantic search pipeline that crawls potential client websites to detect employer-provided childcare benefits.

## Architecture

Two-stage detection per site:

1. **Stage 1 (Exact Match)**: During crawl, check every page for keyword substrings. If found anywhere on the site, skip Stage 2.
2. **Stage 2 (Vector Search)**: If no exact match, chunk all page text, compute embeddings via Vertex AI, and run in-memory cosine similarity. Borderline results are confirmed by Gemini LLM.

## Stack

- **Crawler**: Custom async BFS with Playwright (JS rendering) + BeautifulSoup (text extraction)
- **Exact Match**: Case-insensitive substring matching against keyword list
- **Embeddings**: Vertex AI `text-embedding-004`
- **Vector Search**: In-memory cosine similarity (NumPy)
- **LLM**: Vertex AI Gemini `gemini-1.5-flash` for borderline RAG
- **Storage**: BigQuery (`bh_scraper` dataset)
- **Deployment**: Cloud Run Jobs (manual trigger)

## Prerequisites

- Python 3.11+
- GCP project `mybrightday-dev` with:
  - BigQuery API enabled
  - Vertex AI API enabled
  - Application Default Credentials configured (`gcloud auth application-default login`)
- Playwright Chromium browser

## Setup

```bash
# Clone and enter project
cd BH-Marketing-Scraper-AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser
playwright install chromium
```

## Configuration

Edit `config/config.yaml` for pipeline settings (thresholds, crawl limits, GCP project).

Edit `config/sites.yaml` to add target sites:

```yaml
sites:
  - url: https://example.com
    name: Example Corp
  - url: https://another-site.com
    name: Another Company
```

## Running Locally

```bash
# Standard run
python -m src.main

# With custom config
python -m src.main --config path/to/config.yaml

# Verbose logging
python -m src.main -v
```

## Deploying to Cloud Run

### One-time setup

```bash
# Set project
gcloud config set project mybrightday-dev

# Create Artifact Registry repo
gcloud artifacts repositories create bh-scraper \
  --repository-format=docker \
  --location=us-east4

# Create service account
gcloud iam service-accounts create bh-scraper-sa \
  --display-name="BH Scraper Service Account"

# Grant roles
gcloud projects add-iam-policy-binding mybrightday-dev \
  --member="serviceAccount:bh-scraper-sa@mybrightday-dev.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding mybrightday-dev \
  --member="serviceAccount:bh-scraper-sa@mybrightday-dev.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### Build and deploy

```bash
# Build and push via Cloud Build
gcloud builds submit --config deployment/cloudbuild.yaml .

# Or build locally
docker build -t us-east4-docker.pkg.dev/mybrightday-dev/bh-scraper/bh-marketing-scraper:latest \
  -f deployment/Dockerfile .
docker push us-east4-docker.pkg.dev/mybrightday-dev/bh-scraper/bh-marketing-scraper:latest

# Deploy Cloud Run Job
gcloud run jobs replace deployment/cloud-run-job.yaml --region=us-east4
```

### Run the job

```bash
gcloud run jobs execute bh-scraper-job --region=us-east4
```

## Project Structure

```
BH-Marketing-Scraper-AI/
├── src/
│   ├── main.py              # CLI entry point
│   ├── orchestrator.py      # Pipeline orchestration
│   ├── config.py            # Config loading
│   ├── models.py            # Shared data models
│   ├── crawler/
│   │   └── crawler.py       # Async BFS crawler with Playwright
│   ├── matching/
│   │   ├── exact.py         # Exact substring matcher
│   │   └── vector.py        # Cosine similarity (NumPy)
│   ├── embeddings/
│   │   ├── client.py        # Vertex AI embedding client
│   │   └── chunker.py       # Text chunking
│   ├── llm/
│   │   ├── client.py        # Gemini LLM client
│   │   └── prompts.py       # Prompt templates
│   └── storage/
│       ├── bigquery_client.py  # BigQuery operations
│       └── schema.py        # Table schemas
├── config/
│   ├── config.yaml          # Pipeline configuration
│   └── sites.yaml           # Target site list
├── deployment/
│   ├── Dockerfile
│   ├── cloudbuild.yaml
│   └── cloud-run-job.yaml
├── tests/
├── requirements.txt
├── project-prompt.md         # Source of truth specification
└── README.md
```

## BigQuery Tables

All stored in dataset `bh_scraper` (project: `mybrightday-dev`, location: `us-east4`):

| Table | Purpose |
|-------|---------|
| `visited_pages` | Crawl metadata per page per run |
| `page_content` | Raw HTML and normalized text |
| `scraper_results` | Final matches (exact, vector, llm_confirmed) |
| `run_metadata` | Run-level status and timing |

## Design Decisions

- **Custom BFS crawler** instead of Scrapy to avoid Twisted reactor complexity while maintaining the same crawl capabilities (link extraction, robots.txt, retries, politeness).
- **In-memory page data** during pipeline processing to avoid BigQuery streaming buffer latency. BigQuery writes happen for persistence after processing.
- **Sequential site processing** ensures each site completes fully before moving to the next, as specified.
