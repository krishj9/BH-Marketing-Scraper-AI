"""BigQuery client for the BH Marketing Scraper.

Handles table creation, data insertion, and queries for crawl data,
page content, scraper results, and run metadata.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from src.storage.schema import TABLE_SCHEMAS

logger = logging.getLogger(__name__)


class BigQueryClient:
    """Client for all BigQuery operations in the scraper pipeline."""

    def __init__(self, project_id: str, dataset_id: str, location: str = "us-east4"):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.location = location
        self.client = bigquery.Client(project=project_id)
        self.dataset_ref = f"{project_id}.{dataset_id}"

    def ensure_tables_exist(self) -> None:
        """Create dataset and all tables if they don't exist."""
        # Create dataset
        dataset = bigquery.Dataset(self.dataset_ref)
        dataset.location = self.location
        try:
            self.client.get_dataset(self.dataset_ref)
            logger.info("Dataset %s already exists", self.dataset_ref)
        except NotFound:
            self.client.create_dataset(dataset)
            logger.info("Created dataset %s", self.dataset_ref)

        # Create tables
        for table_name, schema in TABLE_SCHEMAS.items():
            table_ref = f"{self.dataset_ref}.{table_name}"
            table = bigquery.Table(table_ref, schema=schema)
            try:
                self.client.get_table(table_ref)
                logger.debug("Table %s already exists", table_ref)
            except NotFound:
                self.client.create_table(table)
                logger.info("Created table %s", table_ref)

    # ── Batch inserts ──────────────────────────────────────────────

    def insert_visited_pages_batch(self, rows: list[dict[str, Any]]) -> None:
        """Batch insert visited page records."""
        if not rows:
            return
        table_ref = f"{self.dataset_ref}.visited_pages"
        errors = self.client.insert_rows_json(table_ref, rows)
        if errors:
            logger.error("BigQuery insert errors (visited_pages): %s", errors)
        else:
            logger.debug("Inserted %d visited_pages rows", len(rows))

    def insert_page_content_batch(self, rows: list[dict[str, Any]]) -> None:
        """Batch insert page content records."""
        if not rows:
            return
        table_ref = f"{self.dataset_ref}.page_content"
        errors = self.client.insert_rows_json(table_ref, rows)
        if errors:
            logger.error("BigQuery insert errors (page_content): %s", errors)
        else:
            logger.debug("Inserted %d page_content rows", len(rows))

    def insert_scraper_results_batch(self, rows: list[dict[str, Any]]) -> None:
        """Batch insert scraper result records."""
        if not rows:
            return
        table_ref = f"{self.dataset_ref}.scraper_results"
        errors = self.client.insert_rows_json(table_ref, rows)
        if errors:
            logger.error("BigQuery insert errors (scraper_results): %s", errors)
        else:
            logger.debug("Inserted %d scraper_results rows", len(rows))

    # ── Single inserts (convenience) ───────────────────────────────

    def insert_scraper_result(
        self,
        run_id: str,
        site_id: str,
        url: str,
        match_type: str,
        matched_keyword: str | None = None,
        similarity_score: float | None = None,
        llm_explanation: str | None = None,
    ) -> None:
        """Insert a single scraper result."""
        row = {
            "run_id": run_id,
            "site_id": site_id,
            "url": url,
            "match_type": match_type,
            "matched_keyword": matched_keyword,
            "similarity_score": similarity_score,
            "llm_explanation": llm_explanation,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.insert_scraper_results_batch([row])

    # ── Run metadata ───────────────────────────────────────────────

    def insert_run_metadata(
        self, run_id: str, started_at: datetime, run_mode: str
    ) -> None:
        """Insert initial run metadata record."""
        table_ref = f"{self.dataset_ref}.run_metadata"
        rows = [
            {
                "run_id": run_id,
                "started_at": started_at.isoformat(),
                "run_mode": run_mode,
            }
        ]
        errors = self.client.insert_rows_json(table_ref, rows)
        if errors:
            logger.error("BigQuery insert errors (run_metadata): %s", errors)

    def update_run_completed(
        self, run_id: str, sites_processed: int, sites_failed: int
    ) -> None:
        """Update run metadata with completion info via DML."""
        table_ref = f"{self.dataset_ref}.run_metadata"
        query = f"""
            UPDATE `{table_ref}`
            SET completed_at = @completed_at,
                sites_processed = @sites_processed,
                sites_failed = @sites_failed
            WHERE run_id = @run_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "completed_at", "TIMESTAMP", datetime.now(timezone.utc)
                ),
                bigquery.ScalarQueryParameter(
                    "sites_processed", "INT64", sites_processed
                ),
                bigquery.ScalarQueryParameter(
                    "sites_failed", "INT64", sites_failed
                ),
                bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
            ]
        )
        self.client.query(query, job_config=job_config).result()
        logger.info("Updated run %s: processed=%d, failed=%d", run_id, sites_processed, sites_failed)

    # ── Queries ────────────────────────────────────────────────────

    def get_page_content(
        self, run_id: str, site_id: str
    ) -> list[dict[str, Any]]:
        """Retrieve all page content for a site in a run."""
        query = f"""
            SELECT url, normalized_text
            FROM `{self.dataset_ref}.page_content`
            WHERE run_id = @run_id AND site_id = @site_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
                bigquery.ScalarQueryParameter("site_id", "STRING", site_id),
            ]
        )
        results = self.client.query(query, job_config=job_config).result()
        return [
            {"url": row.url, "normalized_text": row.normalized_text}
            for row in results
        ]

    def delete_visited_pages(self, run_id: str, site_id: str) -> None:
        """Archive/delete visited pages for a site after processing."""
        query = f"""
            DELETE FROM `{self.dataset_ref}.visited_pages`
            WHERE run_id = @run_id AND site_id = @site_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
                bigquery.ScalarQueryParameter("site_id", "STRING", site_id),
            ]
        )
        self.client.query(query, job_config=job_config).result()
        logger.info("Archived visited pages for %s in run %s", site_id, run_id)
