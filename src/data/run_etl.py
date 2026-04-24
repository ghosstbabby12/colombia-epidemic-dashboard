"""
Complete ETL orchestrator.

Steps:
1. Download raw files with Scrapy.
2. Clean and normalize datasets.
3. Generate clean CSV and Parquet files.
4. Create PostgreSQL database if needed.
5. Load dashboard and DWH tables.

Usage:
    python src/data/run_etl.py
    python src/data/run_etl.py --year 2024 --force
    python -m src.data.run_etl --year 2024 --force
"""

import argparse
import logging

from src.data.ingestion import run as run_ingestion
from src.data.preprocessing import run as run_preprocessing
from src.data.load_to_postgres import run as run_postgres_load


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger(__name__)


def run_full_etl(year: int = 2024, force_download: bool = False) -> None:
    """Runs the complete ETL pipeline."""

    log.info("====================================================")
    log.info("STARTING COLOMBIA EPIDEMIC DASHBOARD ETL")
    log.info("====================================================")

    log.info("STEP 1/3 - Downloading raw datasets with Scrapy")
    run_ingestion(year=year, force_download=force_download)

    log.info("STEP 2/3 - Cleaning and normalizing datasets")
    run_preprocessing()

    log.info("STEP 3/3 - Loading datasets into PostgreSQL")
    run_postgres_load()

    log.info("====================================================")
    log.info("ETL COMPLETED SUCCESSFULLY")
    log.info("====================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full ETL pipeline for Colombia Epidemic Dashboard.",
    )

    parser.add_argument(
        "--year",
        "--anio",
        type=int,
        default=2024,
        help="Dataset year.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force raw dataset re-download.",
    )

    args = parser.parse_args()

    run_full_etl(
        year=args.year,
        force_download=args.force,
    )