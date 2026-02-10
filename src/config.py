"""Configuration loading for the BH Marketing Scraper."""

import logging
import os
from typing import Any

import yaml

from src.models import SiteConfig

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = "config/config.yaml"


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Path to config YAML. Falls back to CONFIG_PATH env var,
                     then to config/config.yaml.

    Returns:
        Merged configuration dictionary.
    """
    path = config_path or os.environ.get("CONFIG_PATH", _DEFAULT_CONFIG_PATH)
    logger.info("Loading config from %s", path)

    with open(path) as f:
        config = yaml.safe_load(f)

    # Environment variable overrides (for Cloud Run deployment)
    if os.environ.get("GCP_PROJECT_ID"):
        config.setdefault("gcp", {})["project_id"] = os.environ["GCP_PROJECT_ID"]
    if os.environ.get("GCP_REGION"):
        config.setdefault("gcp", {})["region"] = os.environ["GCP_REGION"]
    if os.environ.get("BIGQUERY_DATASET"):
        config.setdefault("gcp", {})["bigquery_dataset"] = os.environ["BIGQUERY_DATASET"]
    if os.environ.get("RUN_MODE"):
        config["run_mode"] = os.environ["RUN_MODE"]

    return config


def load_sites(config: dict[str, Any]) -> list[SiteConfig]:
    """Load target sites from the site list file.

    Args:
        config: Application configuration dict.

    Returns:
        List of SiteConfig objects.
    """
    site_list_path = config.get("site_list_path", "config/sites.yaml")
    logger.info("Loading sites from %s", site_list_path)

    with open(site_list_path) as f:
        data = yaml.safe_load(f)

    sites = []
    for entry in data.get("sites", []):
        if not entry.get("url"):
            logger.warning("Skipping site entry with no URL: %s", entry)
            continue
        sites.append(
            SiteConfig(
                url=entry["url"],
                name=entry.get("name"),
            )
        )

    logger.info("Loaded %d sites", len(sites))
    return sites
