"""
Loads clean epidemiological datasets into Supabase using Supabase API key.

This script uses:
- SUPABASE_URL
- SUPABASE_SERVICE_ROLE_KEY

Important:
    Do not use the service role key in the frontend.
"""

import logging
import math
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client


# -----------------------------------------------------------------------------
# Environment and logging
# -----------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# -----------------------------------------------------------------------------
# Supabase config
# -----------------------------------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
LOAD_MODE = os.getenv("LOAD_MODE", "replace").lower().strip()

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env"
    )


def get_client() -> Client:
    """
    Creates Supabase client.

    Comentario:
    Este cliente usa service role key, por eso solo debe ejecutarse en servidor
    o localmente, nunca en navegador.
    """

    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def read_processed_dataset(base_name: str) -> pd.DataFrame:
    """Reads processed dataset from Parquet or CSV."""

    parquet_path = PROCESSED_DIR / f"{base_name}.parquet"
    csv_path = PROCESSED_DIR / f"{base_name}.csv"

    if parquet_path.exists():
        log.info("Reading Parquet: %s", parquet_path.name)
        return pd.read_parquet(parquet_path)

    if csv_path.exists():
        log.info("Reading CSV: %s", csv_path.name)
        return pd.read_csv(csv_path, dtype=str, low_memory=False)

    raise FileNotFoundError(
        f"Processed dataset not found: {parquet_path.name} or {csv_path.name}"
    )


def clean_json_value(value: Any) -> Any:
    """
    Converts Python, NumPy and Pandas values to strict JSON-safe values.

    Comentario:
    Supabase/PostgREST no acepta NaN, inf, -inf ni NaT en JSON.
    Esta función convierte esos valores a None antes de enviar los lotes.
    """

    if value is None:
        return None

    if value is pd.NA:
        return None

    if value is pd.NaT:
        return None

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        return pd.Timestamp(value).isoformat()

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Decimal):
        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            return None

        return numeric_value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            return None

        return numeric_value

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        return value

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Converts DataFrame to strict JSON-safe Supabase records.

    Comentario:
    Convierte NaN, NaT, pd.NA, numpy.nan, inf y -inf a None.
    Esto corrige el error:
    ValueError: Out of range float values are not JSON compliant: nan
    """

    df = df.copy()

    # Comentario:
    # Se eliminan valores infinitos antes de convertir a objetos JSON.
    df = df.replace([np.inf, -np.inf], np.nan)

    # Comentario:
    # Se fuerza object para que pandas permita reemplazar correctamente NaN por None.
    df = df.astype(object)

    # Comentario:
    # Esta línea cubre la mayoría de NaN/NaT/pd.NA.
    df = df.where(pd.notnull(df), None)

    raw_records = df.to_dict(orient="records")

    safe_records = [
        {
            key: clean_json_value(value)
            for key, value in record.items()
        }
        for record in raw_records
    ]

    return safe_records


def insert_records(
    supabase: Client,
    table_name: str,
    records: list[dict[str, Any]],
    chunk_size: int = 500,
) -> None:
    """
    Inserts records into Supabase table in chunks.

    Comentario:
    Supabase API trabaja mejor con lotes moderados.
    Si un lote falla, se informa el rango exacto para depurar.
    """

    total = len(records)

    if total == 0:
        log.warning("No records to insert into %s", table_name)
        return

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = records[start:end]

        try:
            supabase.table(table_name).insert(
                chunk,
                returning="minimal",
            ).execute()

            log.info(
                "Inserted %s/%s rows into %s",
                f"{end:,}",
                f"{total:,}",
                table_name,
            )

        except Exception as exc:
            log.error(
                "Failed inserting rows %s to %s into %s",
                f"{start:,}",
                f"{end:,}",
                table_name,
            )

            sample = chunk[0] if chunk else {}
            log.error("Sample row from failed chunk: %s", sample)

            raise exc


def insert_dataframe(
    supabase: Client,
    table_name: str,
    df: pd.DataFrame,
    chunk_size: int = 500,
) -> None:
    """Converts and inserts dataframe."""

    log.info(
        "Preparing %s rows for table %s",
        f"{len(df):,}",
        table_name,
    )

    records = dataframe_to_records(df)

    insert_records(
        supabase=supabase,
        table_name=table_name,
        records=records,
        chunk_size=chunk_size,
    )


def reset_tables(supabase: Client) -> None:
    """Resets dashboard tables through secure RPC."""

    if LOAD_MODE != "replace":
        log.info("LOAD_MODE=%s. Skipping reset.", LOAD_MODE)
        return

    log.info("Resetting Supabase tables through RPC")
    supabase.rpc("reset_epidemiology_tables").execute()
    log.info("Tables reset completed")


# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------

def load_dashboard_tables(supabase: Client) -> tuple[int, int]:
    """Loads clean and aggregated datasets."""

    clean_df = read_processed_dataset("sivigila_clean")
    agg_df = read_processed_dataset("sivigila_agg")

    insert_dataframe(
        supabase=supabase,
        table_name="casos_individuales",
        df=clean_df,
        chunk_size=500,
    )

    insert_dataframe(
        supabase=supabase,
        table_name="casos_agregados",
        df=agg_df,
        chunk_size=500,
    )

    return len(clean_df), len(agg_df)


def insert_metadata(
    supabase: Client,
    clean_rows: int,
    aggregated_rows: int,
    prediction_rows: int = 0,
) -> None:
    """Inserts ETL metadata."""

    record = {
        "source": "sivigila_scrapy_etl_supabase_api",
        "clean_rows": int(clean_rows),
        "aggregated_rows": int(aggregated_rows),
        "prediction_rows": int(prediction_rows),
    }

    safe_record = {
        key: clean_json_value(value)
        for key, value in record.items()
    }

    supabase.table("etl_load_metadata").insert(safe_record).execute()

    log.info("ETL metadata inserted successfully")


def run() -> None:
    """Runs Supabase API load."""

    log.info("Starting Supabase API load")

    supabase = get_client()

    reset_tables(supabase)

    clean_rows, aggregated_rows = load_dashboard_tables(supabase)

    insert_metadata(
        supabase=supabase,
        clean_rows=clean_rows,
        aggregated_rows=aggregated_rows,
    )

    log.info("Supabase API load completed successfully")
    log.info("Clean rows: %s", f"{clean_rows:,}")
    log.info("Aggregated rows: %s", f"{aggregated_rows:,}")


if __name__ == "__main__":
    run()