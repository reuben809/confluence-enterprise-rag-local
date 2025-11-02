#!/usr/bin/env python3
"""
Main Ingestion Service Entrypoint
Selects and runs the appropriate data pipeline based on configuration.
"""

import logging
import sys
from src.config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "service": "run_ingestion"}',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def main():
    """
    Selects and runs the data ingestion pipeline.
    """
    mode = config.INGEST_MODE
    logger.info(f"Starting ingestion service in '{mode}' mode.")

    if mode == "mock":
        from src.mock_pipeline import run_pipeline
        logger.info("Running mock data pipeline...")
        run_pipeline()

    elif mode == "confluence":
        from src.data_pipeline import run_pipeline
        logger.info("Running Confluence data pipeline...")
        run_pipeline()

    elif mode == "online":
        from src.online_pipeline import run_pipeline
        logger.info("Running online data pipeline...")
        run_pipeline()

    else:
        logger.error(f"Unknown INGEST_MODE: '{mode}'. Must be 'mock', 'confluence', or 'online'.")
        sys.exit(1)

    logger.info(f"Ingestion service ({mode} mode) completed.")


if __name__ == "__main__":
    main()