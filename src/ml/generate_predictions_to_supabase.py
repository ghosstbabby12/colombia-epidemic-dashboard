"""
Generates ML predictions and uploads them to Supabase.

This allows the Vercel frontend to read predictions directly from Supabase
without running Python/scikit-learn in the browser.
"""

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

from src.ml.model import predict_outbreaks


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
# Supabase
# -----------------------------------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")


def get_client() -> Client:
    """Creates Supabase service client."""

    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_clean_dataset() -> pd.DataFrame:
    """Loads clean epidemiological dataset."""

    parquet_path = PROCESSED_DIR / "sivigila_clean.parquet"
    csv_path = PROCESSED_DIR / "sivigila_clean.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    if csv_path.exists():
        return pd.read_csv(csv_path, dtype=str, low_memory=False)

    raise FileNotFoundError("Run preprocessing first.")


def normalize_text(value: Any) -> str:
    """Normalizes text for stable query keys."""

    return str(value).strip().upper()


def build_prediction_inputs(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Builds disease + department + city + month combinations."""

    base = clean_df[
        [
            "enfermedad",
            "Departamento_ocurrencia",
            "Municipio_ocurrencia",
        ]
    ].copy()

    base = base.dropna()
    base = base[
        (base["Departamento_ocurrencia"] != "SIN_DATO")
        & (base["Municipio_ocurrencia"] != "SIN_DATO")
    ]

    base = base.drop_duplicates()

    base["disease"] = base["enfermedad"].astype(str).str.title()
    base["department"] = base["Departamento_ocurrencia"].apply(normalize_text)
    base["city"] = base["Municipio_ocurrencia"].apply(normalize_text)

    base = base[["disease", "department", "city"]].drop_duplicates()

    months = pd.DataFrame({"month": list(range(1, 13))})

    base["_tmp_key"] = 1
    months["_tmp_key"] = 1

    prediction_inputs = (
        base.merge(months, on="_tmp_key", how="outer")
        .drop(columns="_tmp_key")
        .sort_values(["disease", "department", "city", "month"])
        .reset_index(drop=True)
    )

    return prediction_inputs


def to_supabase_record(prediction: dict[str, Any]) -> dict[str, Any]:
    """Maps model prediction response to Supabase row."""

    validation = prediction.get("model_validation_metrics", {}) or {}

    return {
        "disease": prediction["disease"],
        "month": int(prediction["month"]),
        "department": normalize_text(prediction["department"]),
        "city": normalize_text(prediction["city"]),
        "estimated_cases": int(prediction["estimated_cases"]),
        "estimated_outbreak_proxy": int(prediction["estimated_outbreak_proxy"]),
        "outbreak_level": prediction["outbreak_level"],
        "model_used": prediction.get("model_used"),
        "thermal_floor": prediction.get("thermal_floor"),
        "avg_temp_c": prediction.get("avg_temp_c"),
        "precipitation_mm": prediction.get("precipitation_mm"),
        "humidity_pct": prediction.get("humidity_pct"),
        "tropical_score": prediction.get("tropical_score"),
        "climate_risk_score": prediction.get("climate_risk_score"),
        "is_rainy_season": prediction.get("is_rainy_season"),
        "is_vector_favorable": prediction.get("is_vector_favorable"),
        "cases_previous_month": prediction.get("cases_previous_month"),
        "cases_previous_2_month_mean": prediction.get("cases_previous_2_month_mean"),
        "cases_previous_3_month_mean": prediction.get("cases_previous_3_month_mean"),
        "validation_r2": validation.get("r2"),
        "validation_mae": validation.get("mae"),
        "validation_wape_pct": validation.get("wape_pct"),
    }


def upsert_records(
    supabase: Client,
    records: list[dict[str, Any]],
    chunk_size: int = 300,
) -> None:
    """Upserts predictions into Supabase."""

    total = len(records)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = records[start:end]

        supabase.table("ml_predictions").upsert(
            chunk,
            on_conflict="disease,month,department,city",
            returning="minimal",
        ).execute()

        log.info("Uploaded predictions %s/%s", f"{end:,}", f"{total:,}")


def run() -> None:
    """Generates and uploads all predictions."""

    supabase = get_client()

    clean_df = load_clean_dataset()
    input_df = build_prediction_inputs(clean_df)

    log.info("Prediction combinations: %s", f"{len(input_df):,}")

    records = []

    for index, row in input_df.iterrows():
        prediction = predict_outbreaks(
            month=int(row["month"]),
            department=row["department"],
            city=row["city"],
            disease=row["disease"],
        )

        records.append(to_supabase_record(prediction))

        if (index + 1) % 100 == 0:
            log.info("Generated %s/%s predictions", index + 1, len(input_df))

    upsert_records(supabase, records)

    supabase.table("etl_load_metadata").insert({
        "source": "ml_predictions_random_forest",
        "clean_rows": 0,
        "aggregated_rows": 0,
        "prediction_rows": len(records),
    }).execute()

    log.info("Prediction generation completed")


if __name__ == "__main__":
    run()