"""BigQuery table schemas for the BH Marketing Scraper.

Matches the schema defined in project-prompt.md BigQuery Schema section.
"""

from google.cloud.bigquery import SchemaField

VISITED_PAGES_SCHEMA = [
    SchemaField("run_id", "STRING", mode="REQUIRED"),
    SchemaField("site_id", "STRING", mode="REQUIRED"),
    SchemaField("url", "STRING", mode="REQUIRED"),
    SchemaField("http_status", "INTEGER"),
    SchemaField("crawl_timestamp", "TIMESTAMP", mode="REQUIRED"),
    SchemaField("blocked_reason", "STRING"),
]

PAGE_CONTENT_SCHEMA = [
    SchemaField("run_id", "STRING", mode="REQUIRED"),
    SchemaField("site_id", "STRING", mode="REQUIRED"),
    SchemaField("url", "STRING", mode="REQUIRED"),
    SchemaField("html", "STRING"),
    SchemaField("normalized_text", "STRING"),
    SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
]

SCRAPER_RESULTS_SCHEMA = [
    SchemaField("run_id", "STRING", mode="REQUIRED"),
    SchemaField("site_id", "STRING", mode="REQUIRED"),
    SchemaField("url", "STRING", mode="REQUIRED"),
    SchemaField("match_type", "STRING", mode="REQUIRED"),
    SchemaField("matched_keyword", "STRING"),
    SchemaField("similarity_score", "FLOAT"),
    SchemaField("llm_explanation", "STRING"),
    SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
]

RUN_METADATA_SCHEMA = [
    SchemaField("run_id", "STRING", mode="REQUIRED"),
    SchemaField("started_at", "TIMESTAMP", mode="REQUIRED"),
    SchemaField("completed_at", "TIMESTAMP"),
    SchemaField("sites_processed", "INTEGER"),
    SchemaField("sites_failed", "INTEGER"),
    SchemaField("run_mode", "STRING"),
]

# Map table names to schemas for easy iteration
TABLE_SCHEMAS = {
    "visited_pages": VISITED_PAGES_SCHEMA,
    "page_content": PAGE_CONTENT_SCHEMA,
    "scraper_results": SCRAPER_RESULTS_SCHEMA,
    "run_metadata": RUN_METADATA_SCHEMA,
}
