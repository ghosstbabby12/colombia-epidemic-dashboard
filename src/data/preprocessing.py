"""
Pipeline ETL: limpieza, transformación e integración de datasets.

Entrada : data/raw/
Salida  : data/processed/features.parquet
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Columnas esperadas en el CSV de SIVIGILA después de la ingesta
SIVIGILA_COLS = {
    "año": "year",
    "semana": "week",
    "departamento": "department",
    "municipio": "municipality",
    "casos": "cases",
    "tipo": "type",
}

CLIMA_COLS = {
    "año": "year",
    "semana_epidemiologica": "week",
    "departamento": "department",
    "temperatura_media": "temp_avg",
    "precipitacion_mm": "precipitation_mm",
    "humedad_relativa": "humidity_pct",
}


# ── Limpieza de datos epidemiológicos ─────────────────────────────────────────

def clean_dengue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in SIVIGILA_COLS.items() if k in df.columns})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["cases"] = pd.to_numeric(df["cases"], errors="coerce").fillna(0).astype(int)
    df["department"] = df["department"].str.upper().str.strip()
    df["municipality"] = df["municipality"].str.upper().str.strip()
    df = df.dropna(subset=["year", "week", "department"])
    df = df[(df["week"].between(1, 52)) & (df["year"] > 2000)]
    return df


def aggregate_by_dept_week(df: pd.DataFrame) -> pd.DataFrame:
    """Suma de casos por departamento + semana epidemiológica."""
    return (
        df.groupby(["year", "week", "department"], as_index=False)
        .agg(total_cases=("cases", "sum"), n_records=("cases", "count"))
    )


# ── Limpieza de datos climáticos ──────────────────────────────────────────────

def clean_clima(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in CLIMA_COLS.items() if k in df.columns})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["department"] = df["department"].str.upper().str.strip()
    for col in ["temp_avg", "precipitation_mm", "humidity_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["year", "week", "department"])


# ── Integración de datasets ───────────────────────────────────────────────────

def merge_datasets(
    df_dengue: pd.DataFrame,
    df_clima: pd.DataFrame,
) -> pd.DataFrame:
    df = df_dengue.merge(
        df_clima,
        on=["year", "week", "department"],
        how="left",
    )
    log.info("Merge completado: %d filas, %d columnas", *df.shape)
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["department", "year", "week"]).copy()

    grp = df.groupby("department")

    # Rezagos temporales de casos (2, 4 y 8 semanas)
    for lag in [2, 4, 8]:
        df[f"cases_lag_{lag}w"] = grp["total_cases"].shift(lag)

    # Media móvil de 4 semanas
    df["cases_ma4"] = (
        grp["total_cases"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    )

    # Componentes temporales
    df["month"] = ((df["week"] - 1) // 4 + 1).clip(1, 12)
    df["is_rainy_season"] = df["month"].isin([4, 5, 6, 10, 11]).astype(int)

    # Interacción clima-dengue
    if "precipitation_mm" in df.columns:
        df["precip_x_humidity"] = (
            df["precipitation_mm"].fillna(0) * df["humidity_pct"].fillna(70)
        )

    # Etiqueta de riesgo (target para clasificación)
    df["risk_label"] = pd.cut(
        df["total_cases"],
        bins=[-1, 50, 200, np.inf],
        labels=["bajo", "medio", "alto"],
    )

    df = df.fillna(0)
    log.info("Feature engineering completado. Shape final: %s", df.shape)
    return df


# ── Pipeline principal ────────────────────────────────────────────────────────

def run(year: int = 2023) -> pd.DataFrame:
    log.info("=== Inicio del pipeline ETL ===")

    dengue_path = RAW_DIR / f"sivigila_dengue_{year}.csv"
    clima_path = RAW_DIR / "ideam_clima_2023.csv"

    if not dengue_path.exists():
        log.error("Archivo no encontrado: %s. Ejecuta primero ingestion.py.", dengue_path)
        raise FileNotFoundError(dengue_path)

    df_dengue_raw = pd.read_csv(dengue_path, dtype=str)
    df_dengue = clean_dengue(df_dengue_raw)
    df_dengue_agg = aggregate_by_dept_week(df_dengue)
    log.info("Dengue limpio: %d filas", len(df_dengue_agg))

    df_clima = pd.DataFrame()
    if clima_path.exists():
        df_clima_raw = pd.read_csv(clima_path)
        df_clima = clean_clima(df_clima_raw)
        log.info("Clima limpio: %d filas", len(df_clima))

    if df_clima.empty:
        df_features = df_dengue_agg
    else:
        df_features = merge_datasets(df_dengue_agg, df_clima)

    df_features = engineer_features(df_features)

    out_path = PROCESSED_DIR / "features.parquet"
    df_features.to_parquet(out_path, index=False)
    log.info("Guardado: %s", out_path)
    log.info("=== ETL completado ===")

    return df_features


if __name__ == "__main__":
    run(year=2023)
