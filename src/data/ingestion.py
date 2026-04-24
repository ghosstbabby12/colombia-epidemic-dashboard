"""
Automatic epidemiological dataset downloader with Scrapy.

This module downloads raw datasets for:
- Dengue
- Chikungunya
- Malaria

Sources:
- Medellín open data CSV files
- INS SIVIGILA microdata Excel files

Usage:
    python src/data/ingestion.py
    python src/data/ingestion.py --year 2024 --force
    python -m src.data.ingestion --year 2024 --force
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import scrapy
from scrapy.crawler import CrawlerProcess


# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Dataset catalog
# -----------------------------------------------------------------------------
# Nota:
# - Se usa Scrapy para descargar directamente los archivos.
# - Los enlaces del portal INS ya están resueltos al archivo real /Microdatos/.
# - Malaria se configura con Datos_2024_490.xlsx, no 460.

DATASET_CATALOG: list[dict[str, Any]] = [
    {
        "disease": "Dengue",
        "slug": "dengue",
        "year": 2024,
        "event_code": 210,
        "source": "medata",
        "url": "http://medata.gov.co/sites/default/files/distribution/1-026-22-000135/sivigila_dengue.csv",
        "filename": "sivigila_dengue_medata_2024.csv",
    },
    {
        "disease": "Dengue",
        "slug": "dengue",
        "year": 2024,
        "event_code": 210,
        "source": "ins_sivigila",
        "url": "https://portalsivigila.ins.gov.co/Microdatos/Datos_2024_210.xlsx",
        "filename": "Datos_2024_210.xlsx",
    },
    {
        "disease": "Chikungunya",
        "slug": "chikungunya",
        "year": 2024,
        "event_code": 217,
        "source": "medata",
        "url": "http://medata.gov.co/sites/default/files/distribution/1-026-22-000130/sivigila_chikungunya.csv",
        "filename": "sivigila_chikungunya_medata_2024.csv",
    },
    {
        "disease": "Chikungunya",
        "slug": "chikungunya",
        "year": 2024,
        "event_code": 217,
        "source": "ins_sivigila",
        "url": "https://portalsivigila.ins.gov.co/Microdatos/Datos_2024_217.xlsx",
        "filename": "Datos_2024_217.xlsx",
    },
    {
        "disease": "Malaria",
        "slug": "malaria",
        "year": 2024,
        "event_code": 490,
        "source": "ins_sivigila",
        "url": "https://portalsivigila.ins.gov.co/Microdatos/Datos_2024_490.xlsx",
        "filename": "Datos_2024_490.xlsx",
    },
]


# -----------------------------------------------------------------------------
# Scrapy spider
# -----------------------------------------------------------------------------

class EpidemiologicalDatasetSpider(scrapy.Spider):
    """
    Spider responsible for downloading raw epidemiological files.

    Comentario:
    Este spider no ejecuta JavaScript. Descarga directamente los archivos finales
    usando los enlaces reales suministrados.
    """

    name = "epidemiological_dataset_downloader"

    custom_settings = {
        "LOG_LEVEL": "INFO",
        "ROBOTSTXT_OBEY": False,
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 3,
        "DOWNLOAD_TIMEOUT": 180,
        "CONCURRENT_REQUESTS": 4,
        "DOWNLOAD_DELAY": 0.5,
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/146.0 Safari/537.36"
        ),
    }

    def __init__(self, year: int = 2024, force_download: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.year = int(year)
        self.force_download = self._to_bool(force_download)
        self.manifest: list[dict[str, Any]] = []

        self.datasets = [
            dataset for dataset in DATASET_CATALOG
            if int(dataset["year"]) == self.year
        ]

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value

        return str(value).lower().strip() in {"1", "true", "yes", "y", "si", "sí"}

    def start_requests(self):
        if not self.datasets:
            log.warning("No datasets configured for year %s", self.year)
            return

        for dataset in self.datasets:
            output_path = RAW_DIR / dataset["filename"]

            if output_path.exists() and output_path.stat().st_size > 1000 and not self.force_download:
                log.info("Cached file found. Skipping download: %s", output_path.name)

                self.manifest.append({
                    "disease": dataset["disease"],
                    "slug": dataset["slug"],
                    "year": dataset["year"],
                    "event_code": dataset["event_code"],
                    "source": dataset["source"],
                    "url": dataset["url"],
                    "filename": dataset["filename"],
                    "path": str(output_path),
                    "status": "cached",
                    "size_bytes": output_path.stat().st_size,
                })
                continue

            log.info("Downloading %s from %s", dataset["filename"], dataset["url"])

            yield scrapy.Request(
                url=dataset["url"],
                callback=self.save_file,
                errback=self.download_error,
                meta={
                    "dataset": dataset,
                    "output_path": output_path,
                },
                dont_filter=True,
            )

    def save_file(self, response: scrapy.http.Response):
        dataset = response.meta["dataset"]
        output_path: Path = response.meta["output_path"]

        content_type = response.headers.get("Content-Type", b"").decode(
            "utf-8",
            errors="ignore",
        )

        body = response.body

        if response.status != 200:
            log.error("HTTP %s while downloading %s", response.status, dataset["url"])
            self._add_manifest_record(dataset, output_path, "failed_http", 0, content_type)
            return

        if not self._is_valid_file(output_path.name, body):
            log.error("Invalid file content received for %s", output_path.name)
            self._add_manifest_record(dataset, output_path, "invalid_content", len(body), content_type)
            return

        output_path.write_bytes(body)

        log.info(
            "Saved file: %s - %.2f MB",
            output_path.name,
            output_path.stat().st_size / (1024 * 1024),
        )

        self._add_manifest_record(
            dataset=dataset,
            output_path=output_path,
            status="downloaded",
            size_bytes=output_path.stat().st_size,
            content_type=content_type,
        )

    def download_error(self, failure):
        request = failure.request
        dataset = request.meta["dataset"]
        output_path = request.meta["output_path"]

        log.error("Download failed for %s: %s", dataset["url"], failure.value)

        self._add_manifest_record(
            dataset=dataset,
            output_path=output_path,
            status="failed_request",
            size_bytes=0,
            content_type="",
        )

    @staticmethod
    def _is_valid_file(filename: str, body: bytes) -> bool:
        """
        Validates basic file signatures.

        Comentario:
        - XLSX files are ZIP-based and usually start with PK.
        - CSV files should not be an HTML error page.
        """

        if len(body) < 100:
            return False

        lower_name = filename.lower()

        if lower_name.endswith(".xlsx"):
            return body[:2] == b"PK"

        if lower_name.endswith(".csv"):
            preview = body[:500].decode("utf-8", errors="ignore").lower()
            if "<html" in preview or "<!doctype html" in preview:
                return False
            return True

        return True

    def _add_manifest_record(
        self,
        dataset: dict[str, Any],
        output_path: Path,
        status: str,
        size_bytes: int,
        content_type: str,
    ) -> None:
        self.manifest.append({
            "disease": dataset["disease"],
            "slug": dataset["slug"],
            "year": dataset["year"],
            "event_code": dataset["event_code"],
            "source": dataset["source"],
            "url": dataset["url"],
            "filename": dataset["filename"],
            "path": str(output_path),
            "status": status,
            "size_bytes": size_bytes,
            "content_type": content_type,
        })

    def closed(self, reason: str):
        manifest_path = RAW_DIR / "download_manifest.csv"

        if self.manifest:
            manifest_df = pd.DataFrame(self.manifest)
            manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
            log.info("Download manifest saved: %s", manifest_path)

        log.info("Spider closed. Reason: %s", reason)


# -----------------------------------------------------------------------------
# Public runner
# -----------------------------------------------------------------------------

def run(year: int = 2024, force_download: bool = False) -> None:
    """
    Runs the Scrapy downloader.

    Comentario:
    Esta función se puede llamar desde el orquestador general del ETL.
    """

    log.info("Starting epidemiological data ingestion with Scrapy")
    log.info("Raw output directory: %s", RAW_DIR)

    process = CrawlerProcess()
    process.crawl(
        EpidemiologicalDatasetSpider,
        year=year,
        force_download=force_download,
    )
    process.start()

    log.info("Ingestion process completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download epidemiological datasets using Scrapy.",
    )

    parser.add_argument(
        "--year",
        "--anio",
        type=int,
        default=2024,
        help="Dataset year to download.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist.",
    )

    args = parser.parse_args()

    run(year=args.year, force_download=args.force)