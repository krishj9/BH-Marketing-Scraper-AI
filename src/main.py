"""CLI entry point for the BH Marketing Scraper pipeline.

Usage:
    python -m src.main [--config path/to/config.yaml] [-v]
"""

import argparse
import asyncio
import logging
import sys

from src.config import load_config
from src.orchestrator import run_pipeline


def main() -> None:
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="BH Marketing Scraper — Hybrid keyword + semantic search pipeline",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("BH Marketing Scraper starting")

    try:
        config = load_config(args.config)
        asyncio.run(run_pipeline(config))
    except FileNotFoundError as e:
        logger.error("Configuration file not found: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
