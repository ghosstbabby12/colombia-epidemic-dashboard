"""
Random Forest model for mosquito-borne disease outbreak prediction in Colombia.

This module trains optimized Random Forest models using the cleaned SIVIGILA dataset.

Main goal:
    Predict estimated outbreak/case volume for a given:
    - Month
    - Department
    - City / Municipality
    - Disease

Generated artifacts:
    src/ml/models/model.pkl
    src/ml/models/model_metrics.json
    data/processed/ml_outbreak_training_dataset.csv

Usage:
    python -m src.ml.model --train

    python -m src.ml.model --predict --month 5 --department NARIÑO --city PASTO --disease Dengue
"""

import argparse
import json
import logging
import math
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "model.pkl"
TRAINING_DATASET_PATH = PROCESSED_DIR / "ml_outbreak_training_dataset.csv"
METRICS_PATH = MODELS_DIR / "model_metrics.json"


# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------

TARGET_COL = "total_cases"

MIN_ROWS_FOR_DISEASE_MODEL = 80

CATEGORICAL_FEATURES = [
    "disease",
    "department",
    "municipality",
    "thermal_floor",
    "is_tropical_climate",
    "is_vector_favorable",
]

NUMERIC_FEATURES = [
    "year",
    "month",
    "month_sin",
    "month_cos",
    "is_rainy_season",
    "avg_temp_c",
    "precipitation_mm",
    "humidity_pct",
    "tropical_score",
    "climate_risk_score",
    "cases_previous_month",
    "cases_previous_2_month_mean",
    "cases_previous_3_month_mean",
    "department_global_mean_cases",
    "municipality_global_mean_cases",
    "department_month_mean_cases",
    "municipality_month_mean_cases",
    "department_disease_mean_cases",
    "municipality_disease_mean_cases",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# -----------------------------------------------------------------------------
# Colombia climate / thermal floor configuration
# -----------------------------------------------------------------------------

DEPARTMENT_CLIMATE_PROFILE = {
    "AMAZONAS": {
        "thermal_floor": "calido",
        "avg_temp_c": 26.5,
        "precipitation_mm": 3200,
        "humidity_pct": 88,
        "tropical_score": 1.00,
    },
    "ANTIOQUIA": {
        "thermal_floor": "mixto",
        "avg_temp_c": 23.0,
        "precipitation_mm": 2400,
        "humidity_pct": 80,
        "tropical_score": 0.75,
    },
    "ARAUCA": {
        "thermal_floor": "calido",
        "avg_temp_c": 27.0,
        "precipitation_mm": 2200,
        "humidity_pct": 78,
        "tropical_score": 0.95,
    },
    "ATLANTICO": {
        "thermal_floor": "calido",
        "avg_temp_c": 28.0,
        "precipitation_mm": 900,
        "humidity_pct": 77,
        "tropical_score": 0.90,
    },
    "BOLIVAR": {
        "thermal_floor": "calido",
        "avg_temp_c": 28.0,
        "precipitation_mm": 1200,
        "humidity_pct": 82,
        "tropical_score": 0.95,
    },
    "BOYACA": {
        "thermal_floor": "frio",
        "avg_temp_c": 16.0,
        "precipitation_mm": 1100,
        "humidity_pct": 70,
        "tropical_score": 0.30,
    },
    "CALDAS": {
        "thermal_floor": "templado",
        "avg_temp_c": 21.0,
        "precipitation_mm": 2300,
        "humidity_pct": 80,
        "tropical_score": 0.65,
    },
    "CAQUETA": {
        "thermal_floor": "calido",
        "avg_temp_c": 26.0,
        "precipitation_mm": 3000,
        "humidity_pct": 86,
        "tropical_score": 1.00,
    },
    "CASANARE": {
        "thermal_floor": "calido",
        "avg_temp_c": 27.0,
        "precipitation_mm": 2500,
        "humidity_pct": 80,
        "tropical_score": 0.95,
    },
    "CAUCA": {
        "thermal_floor": "mixto",
        "avg_temp_c": 21.0,
        "precipitation_mm": 1900,
        "humidity_pct": 78,
        "tropical_score": 0.65,
    },
    "CESAR": {
        "thermal_floor": "calido",
        "avg_temp_c": 29.0,
        "precipitation_mm": 1100,
        "humidity_pct": 75,
        "tropical_score": 0.90,
    },
    "CHOCO": {
        "thermal_floor": "calido",
        "avg_temp_c": 27.0,
        "precipitation_mm": 8000,
        "humidity_pct": 90,
        "tropical_score": 1.00,
    },
    "CORDOBA": {
        "thermal_floor": "calido",
        "avg_temp_c": 28.0,
        "precipitation_mm": 1400,
        "humidity_pct": 81,
        "tropical_score": 0.95,
    },
    "CUNDINAMARCA": {
        "thermal_floor": "mixto",
        "avg_temp_c": 19.0,
        "precipitation_mm": 1200,
        "humidity_pct": 72,
        "tropical_score": 0.50,
    },
    "GUAINIA": {
        "thermal_floor": "calido",
        "avg_temp_c": 27.0,
        "precipitation_mm": 3000,
        "humidity_pct": 87,
        "tropical_score": 1.00,
    },
    "GUAVIARE": {
        "thermal_floor": "calido",
        "avg_temp_c": 26.5,
        "precipitation_mm": 2800,
        "humidity_pct": 85,
        "tropical_score": 1.00,
    },
    "HUILA": {
        "thermal_floor": "calido",
        "avg_temp_c": 25.0,
        "precipitation_mm": 1300,
        "humidity_pct": 70,
        "tropical_score": 0.80,
    },
    "LA GUAJIRA": {
        "thermal_floor": "calido",
        "avg_temp_c": 29.5,
        "precipitation_mm": 500,
        "humidity_pct": 65,
        "tropical_score": 0.75,
    },
    "MAGDALENA": {
        "thermal_floor": "calido",
        "avg_temp_c": 28.5,
        "precipitation_mm": 1000,
        "humidity_pct": 76,
        "tropical_score": 0.90,
    },
    "META": {
        "thermal_floor": "calido",
        "avg_temp_c": 26.0,
        "precipitation_mm": 3200,
        "humidity_pct": 82,
        "tropical_score": 1.00,
    },
    "NARIÑO": {
        "thermal_floor": "mixto",
        "avg_temp_c": 18.0,
        "precipitation_mm": 2200,
        "humidity_pct": 81,
        "tropical_score": 0.55,
    },
    "NORTE SANTANDER": {
        "thermal_floor": "calido",
        "avg_temp_c": 24.5,
        "precipitation_mm": 1300,
        "humidity_pct": 72,
        "tropical_score": 0.75,
    },
    "PUTUMAYO": {
        "thermal_floor": "calido",
        "avg_temp_c": 25.0,
        "precipitation_mm": 3500,
        "humidity_pct": 86,
        "tropical_score": 1.00,
    },
    "QUINDIO": {
        "thermal_floor": "templado",
        "avg_temp_c": 21.0,
        "precipitation_mm": 2200,
        "humidity_pct": 80,
        "tropical_score": 0.65,
    },
    "RISARALDA": {
        "thermal_floor": "templado",
        "avg_temp_c": 21.0,
        "precipitation_mm": 2600,
        "humidity_pct": 83,
        "tropical_score": 0.70,
    },
    "SAN ANDRES": {
        "thermal_floor": "insular_calido",
        "avg_temp_c": 27.5,
        "precipitation_mm": 1900,
        "humidity_pct": 82,
        "tropical_score": 0.95,
    },
    "SANTANDER": {
        "thermal_floor": "calido",
        "avg_temp_c": 24.0,
        "precipitation_mm": 1900,
        "humidity_pct": 74,
        "tropical_score": 0.80,
    },
    "SUCRE": {
        "thermal_floor": "calido",
        "avg_temp_c": 28.5,
        "precipitation_mm": 1100,
        "humidity_pct": 79,
        "tropical_score": 0.90,
    },
    "TOLIMA": {
        "thermal_floor": "calido",
        "avg_temp_c": 26.0,
        "precipitation_mm": 1400,
        "humidity_pct": 70,
        "tropical_score": 0.80,
    },
    "VALLE": {
        "thermal_floor": "calido",
        "avg_temp_c": 24.0,
        "precipitation_mm": 1600,
        "humidity_pct": 78,
        "tropical_score": 0.85,
    },
    "VAUPES": {
        "thermal_floor": "calido",
        "avg_temp_c": 26.5,
        "precipitation_mm": 3100,
        "humidity_pct": 88,
        "tropical_score": 1.00,
    },
    "VICHADA": {
        "thermal_floor": "calido",
        "avg_temp_c": 27.0,
        "precipitation_mm": 2400,
        "humidity_pct": 80,
        "tropical_score": 0.95,
    },
    "BOGOTA": {
        "thermal_floor": "frio",
        "avg_temp_c": 14.0,
        "precipitation_mm": 1000,
        "humidity_pct": 75,
        "tropical_score": 0.20,
    },
    "BOGOTA D.C.": {
        "thermal_floor": "frio",
        "avg_temp_c": 14.0,
        "precipitation_mm": 1000,
        "humidity_pct": 75,
        "tropical_score": 0.20,
    },
}

DEFAULT_CLIMATE_PROFILE = {
    "thermal_floor": "mixto",
    "avg_temp_c": 24.0,
    "precipitation_mm": 1600,
    "humidity_pct": 76,
    "tropical_score": 0.70,
}


# -----------------------------------------------------------------------------
# Text normalization
# -----------------------------------------------------------------------------

def normalize_name(value: Any) -> str:
    """Normalizes text for department and municipality comparison."""

    if pd.isna(value):
        return "SIN_DATO"

    text = str(value).strip()

    if not text:
        return "SIN_DATO"

    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.upper()
    text = " ".join(text.split())

    return text


def normalize_disease(value: Any) -> str:
    """Normalizes disease names."""

    text = normalize_name(value)

    disease_map = {
        "DENGUE": "Dengue",
        "CHIKUNGUNYA": "Chikungunya",
        "CHIKUNGUYA": "Chikungunya",
        "MALARIA": "Malaria",
        "MALARIA VIVAX": "Malaria",
    }

    return disease_map.get(text, str(value).strip().title())


# -----------------------------------------------------------------------------
# Loading clean dataset
# -----------------------------------------------------------------------------

def load_clean_dataset() -> pd.DataFrame:
    """Loads the clean dataset generated by preprocessing.py."""

    parquet_path = PROCESSED_DIR / "sivigila_clean.parquet"
    csv_path = PROCESSED_DIR / "sivigila_clean.csv"

    if parquet_path.exists():
        log.info("Loading clean dataset from %s", parquet_path)
        return pd.read_parquet(parquet_path)

    if csv_path.exists():
        log.info("Loading clean dataset from %s", csv_path)
        return pd.read_csv(csv_path, dtype=str, low_memory=False)

    raise FileNotFoundError(
        "Clean dataset not found. Run preprocessing first: python -m src.data.preprocessing"
    )


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

def get_climate_profile(department: Any) -> dict[str, Any]:
    """Returns climate profile for a department."""

    department_norm = normalize_name(department)
    return DEPARTMENT_CLIMATE_PROFILE.get(department_norm, DEFAULT_CLIMATE_PROFILE)


def is_rainy_month(month: int) -> int:
    """Marks commonly rainy months."""

    return int(month in {4, 5, 10, 11})


def compute_climate_risk_score(
    avg_temp_c: float,
    precipitation_mm: float,
    humidity_pct: float,
    tropical_score: float,
    is_rainy_season: int,
) -> float:
    """Computes synthetic climate risk score."""

    temp_component = max(0.0, min(1.0, (avg_temp_c - 14.0) / 16.0))
    rain_component = max(0.0, min(1.0, precipitation_mm / 3000.0))
    humidity_component = max(0.0, min(1.0, humidity_pct / 100.0))

    score = (
        0.35 * temp_component
        + 0.25 * rain_component
        + 0.20 * humidity_component
        + 0.15 * tropical_score
        + 0.05 * is_rainy_season
    )

    return round(score, 4)


def get_outbreak_level(predicted_cases: float) -> str:
    """Converts predicted cases to an outbreak level."""

    if predicted_cases < 5:
        return "Sin brote relevante"

    if predicted_cases < 25:
        return "Brote bajo"

    if predicted_cases < 75:
        return "Brote medio"

    return "Brote alto"


def get_month_from_row(row: pd.Series) -> Any:
    """Gets month from notification date or epidemiological week."""

    if pd.notna(row.get("FEC_NOT")):
        parsed_date = pd.to_datetime(row.get("FEC_NOT"), errors="coerce")

        if pd.notna(parsed_date):
            return int(parsed_date.month)

    try:
        year = int(row.get("ANO"))
        week = int(row.get("SEMANA"))

        return int(pd.Timestamp.fromisocalendar(year, week, 1).month)
    except Exception:
        return pd.NA


def add_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds thermal floor and climate variables."""

    df = df.copy()

    profiles = df["department"].apply(get_climate_profile)

    df["thermal_floor"] = profiles.apply(lambda profile: profile["thermal_floor"])
    df["avg_temp_c"] = profiles.apply(lambda profile: profile["avg_temp_c"])
    df["precipitation_mm"] = profiles.apply(lambda profile: profile["precipitation_mm"])
    df["humidity_pct"] = profiles.apply(lambda profile: profile["humidity_pct"])
    df["tropical_score"] = profiles.apply(lambda profile: profile["tropical_score"])

    df["is_rainy_season"] = df["month"].apply(lambda month: is_rainy_month(int(month)))

    df["is_tropical_climate"] = np.where(
        df["tropical_score"] >= 0.70,
        "SI",
        "NO",
    )

    df["is_vector_favorable"] = np.where(
        (df["avg_temp_c"] >= 22)
        & (df["humidity_pct"] >= 70)
        & (df["tropical_score"] >= 0.65),
        "SI",
        "NO",
    )

    df["climate_risk_score"] = df.apply(
        lambda row: compute_climate_risk_score(
            avg_temp_c=float(row["avg_temp_c"]),
            precipitation_mm=float(row["precipitation_mm"]),
            humidity_pct=float(row["humidity_pct"]),
            tropical_score=float(row["tropical_score"]),
            is_rainy_season=int(row["is_rainy_season"]),
        ),
        axis=1,
    )

    df["month_sin"] = np.sin(2 * np.pi * df["month"].astype(float) / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"].astype(float) / 12)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds previous-month features.

    Comentario:
    Estas variables mejoran bastante el rendimiento porque los brotes suelen
    tener continuidad temporal.
    """

    df = df.copy()

    sort_cols = ["disease", "department", "municipality", "year", "month"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    group_cols = ["disease", "department", "municipality"]

    df["cases_previous_month"] = (
        df.groupby(group_cols)[TARGET_COL]
        .shift(1)
        .fillna(0)
        .astype(float)
    )

    df["cases_previous_2_month_mean"] = (
        df.groupby(group_cols)[TARGET_COL]
        .transform(lambda series: series.shift(1).rolling(2, min_periods=1).mean())
        .fillna(0)
        .astype(float)
    )

    df["cases_previous_3_month_mean"] = (
        df.groupby(group_cols)[TARGET_COL]
        .transform(lambda series: series.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0)
        .astype(float)
    )

    return df


def build_training_dataset(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds balanced monthly model training dataset from individual clean cases.

    Improvement:
        The old version only trained with months where cases existed.
        This version creates a complete 12-month panel per disease + municipality,
        filling missing months with 0 cases.
    """

    df = clean_df.copy()

    required_columns = [
        "enfermedad",
        "ANO",
        "SEMANA",
        "Departamento_ocurrencia",
        "Municipio_ocurrencia",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column missing from clean dataset: {column}")

    df["disease"] = df["enfermedad"].apply(normalize_disease)
    df["year"] = pd.to_numeric(df["ANO"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["SEMANA"], errors="coerce").astype("Int64")
    df["department"] = df["Departamento_ocurrencia"].apply(normalize_name)
    df["municipality"] = df["Municipio_ocurrencia"].apply(normalize_name)

    if "FEC_NOT" in df.columns:
        df["FEC_NOT"] = pd.to_datetime(df["FEC_NOT"], errors="coerce")
    else:
        df["FEC_NOT"] = pd.NaT

    df["month"] = df.apply(get_month_from_row, axis=1)
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["year", "month", "department", "municipality", "disease"])

    df = df[
        (df["month"] >= 1)
        & (df["month"] <= 12)
        & (df["department"] != "SIN_DATO")
        & (df["municipality"] != "SIN_DATO")
    ].copy()

    if df.empty:
        raise ValueError("No valid rows available to build the ML training dataset.")

    positive_counts = (
        df.groupby(
            ["disease", "year", "month", "department", "municipality"],
            dropna=False,
            observed=True,
        )
        .size()
        .reset_index(name=TARGET_COL)
    )

    entity_panel = (
        positive_counts[["disease", "year", "department", "municipality"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    months = pd.DataFrame({"month": list(range(1, 13))})

    entity_panel["_tmp_key"] = 1
    months["_tmp_key"] = 1

    full_panel = entity_panel.merge(months, on="_tmp_key", how="outer").drop(columns="_tmp_key")

    grouped = full_panel.merge(
        positive_counts,
        on=["disease", "year", "month", "department", "municipality"],
        how="left",
    )

    grouped[TARGET_COL] = grouped[TARGET_COL].fillna(0).astype(float)

    grouped = add_climate_features(grouped)
    grouped = add_temporal_features(grouped)

    grouped = grouped.sort_values(
        ["disease", "department", "municipality", "year", "month"],
        ascending=True,
    ).reset_index(drop=True)

    grouped.to_csv(TRAINING_DATASET_PATH, index=False, encoding="utf-8-sig")

    log.info("ML training dataset saved: %s", TRAINING_DATASET_PATH)
    log.info("ML training dataset rows: %s", f"{len(grouped):,}")

    return grouped


# -----------------------------------------------------------------------------
# Lookup and prior features
# -----------------------------------------------------------------------------

def make_key(*values: Any) -> str:
    """Creates a stable lookup key."""

    return "||".join(normalize_name(value) for value in values)


def fit_prior_maps(df: pd.DataFrame) -> dict[str, Any]:
    """
    Fits historical prior maps using only a given dataset split.

    Comentario:
    Esto evita fuga de información en evaluación, porque los promedios del test
    no se usan para construir las variables del test.
    """

    global_mean = float(df[TARGET_COL].mean()) if len(df) else 0.0

    priors: dict[str, Any] = {
        "global_mean_cases": global_mean,
        "department_global_mean_cases": {},
        "municipality_global_mean_cases": {},
        "department_month_mean_cases": {},
        "municipality_month_mean_cases": {},
        "department_disease_mean_cases": {},
        "municipality_disease_mean_cases": {},
        "monthly_cases": {},
    }

    for department, value in df.groupby("department")[TARGET_COL].mean().items():
        priors["department_global_mean_cases"][make_key(department)] = float(value)

    for keys, value in df.groupby(["department", "municipality"])[TARGET_COL].mean().items():
        department, municipality = keys
        priors["municipality_global_mean_cases"][make_key(department, municipality)] = float(value)

    for keys, value in df.groupby(["department", "month"])[TARGET_COL].mean().items():
        department, month = keys
        priors["department_month_mean_cases"][make_key(department, month)] = float(value)

    for keys, value in df.groupby(["department", "municipality", "month"])[TARGET_COL].mean().items():
        department, municipality, month = keys
        priors["municipality_month_mean_cases"][make_key(department, municipality, month)] = float(value)

    for keys, value in df.groupby(["disease", "department"])[TARGET_COL].mean().items():
        disease, department = keys
        priors["department_disease_mean_cases"][make_key(disease, department)] = float(value)

    for keys, value in df.groupby(["disease", "department", "municipality"])[TARGET_COL].mean().items():
        disease, department, municipality = keys
        priors["municipality_disease_mean_cases"][make_key(disease, department, municipality)] = float(value)

    for keys, value in df.groupby(["disease", "department", "municipality", "month"])[TARGET_COL].mean().items():
        disease, department, municipality, month = keys
        priors["monthly_cases"][make_key(disease, department, municipality, month)] = float(value)

    return priors


def lookup_value(
    priors: dict[str, Any],
    lookup_name: str,
    key: str,
    default: float,
) -> float:
    """Safely reads lookup value."""

    value = priors.get(lookup_name, {}).get(key)

    if value is None:
        return float(default)

    return float(value)


def apply_prior_features(df: pd.DataFrame, priors: dict[str, Any]) -> pd.DataFrame:
    """Applies historical prior features to a dataframe."""

    df = df.copy()

    global_mean = float(priors.get("global_mean_cases", 0.0))

    df["department_global_mean_cases"] = df.apply(
        lambda row: lookup_value(
            priors,
            "department_global_mean_cases",
            make_key(row["department"]),
            global_mean,
        ),
        axis=1,
    )

    df["municipality_global_mean_cases"] = df.apply(
        lambda row: lookup_value(
            priors,
            "municipality_global_mean_cases",
            make_key(row["department"], row["municipality"]),
            row["department_global_mean_cases"],
        ),
        axis=1,
    )

    df["department_month_mean_cases"] = df.apply(
        lambda row: lookup_value(
            priors,
            "department_month_mean_cases",
            make_key(row["department"], row["month"]),
            row["department_global_mean_cases"],
        ),
        axis=1,
    )

    df["municipality_month_mean_cases"] = df.apply(
        lambda row: lookup_value(
            priors,
            "municipality_month_mean_cases",
            make_key(row["department"], row["municipality"], row["month"]),
            row["municipality_global_mean_cases"],
        ),
        axis=1,
    )

    df["department_disease_mean_cases"] = df.apply(
        lambda row: lookup_value(
            priors,
            "department_disease_mean_cases",
            make_key(row["disease"], row["department"]),
            row["department_global_mean_cases"],
        ),
        axis=1,
    )

    df["municipality_disease_mean_cases"] = df.apply(
        lambda row: lookup_value(
            priors,
            "municipality_disease_mean_cases",
            make_key(row["disease"], row["department"], row["municipality"]),
            row["municipality_global_mean_cases"],
        ),
        axis=1,
    )

    return df


def get_previous_month_value(
    priors: dict[str, Any],
    disease: str,
    department: str,
    municipality: str,
    month: int,
    fallback: float,
) -> float:
    """Gets historical value for a previous month."""

    fixed_month = int(month)

    if fixed_month < 1:
        fixed_month = 12

    return lookup_value(
        priors,
        "monthly_cases",
        make_key(disease, department, municipality, fixed_month),
        fallback,
    )


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------

def create_one_hot_encoder() -> OneHotEncoder:
    """Creates OneHotEncoder compatible with different sklearn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_model_pipeline(params: dict[str, Any]) -> Pipeline:
    """Builds preprocessing + Random Forest pipeline."""

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", create_one_hot_encoder()),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 800),
        max_depth=params.get("max_depth", None),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=params.get("max_features", "sqrt"),
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def transform_target(y: np.ndarray, method: str) -> np.ndarray:
    """Transforms regression target."""

    if method == "log1p":
        return np.log1p(y)

    if method == "sqrt":
        return np.sqrt(y)

    return y


def inverse_transform_target(y: np.ndarray, method: str) -> np.ndarray:
    """Inverts target transformation."""

    if method == "log1p":
        return np.expm1(y)

    if method == "sqrt":
        return np.square(y)

    return y


def build_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Builds sample weights to give more importance to outbreak rows.

    Comentario:
    El modelo anterior aprendía mucho de los casos bajos y poco de los brotes altos.
    """

    return 1.0 + np.log1p(y)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Evaluates regression performance with robust metrics."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    non_zero_mask = y_true > 0

    if non_zero_mask.any():
        mape = np.mean(
            np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
        ) * 100
    else:
        mape = 0.0

    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        wape = 0.0
    else:
        wape = np.sum(np.abs(y_true - y_pred)) / denominator * 100

    smape_denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_denominator = np.where(smape_denominator == 0, 1, smape_denominator)
    smape = np.mean(np.abs(y_true - y_pred) / smape_denominator) * 100

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "mape_non_zero_pct": round(float(mape), 4),
        "wape_pct": round(float(wape), 4),
        "smape_pct": round(float(smape), 4),
    }


def get_candidate_configs() -> list[dict[str, Any]]:
    """Returns candidate Random Forest configurations."""

    return [
        {
            "name": "rf_balanced_deep_log",
            "target_transform": "log1p",
            "params": {
                "n_estimators": 900,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
        },
        {
            "name": "rf_stable_log",
            "target_transform": "log1p",
            "params": {
                "n_estimators": 800,
                "max_depth": 26,
                "min_samples_split": 3,
                "min_samples_leaf": 1,
                "max_features": 0.75,
            },
        },
        {
            "name": "rf_regularized_log",
            "target_transform": "log1p",
            "params": {
                "n_estimators": 700,
                "max_depth": 20,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
        },
        {
            "name": "rf_balanced_sqrt",
            "target_transform": "sqrt",
            "params": {
                "n_estimators": 800,
                "max_depth": 24,
                "min_samples_split": 3,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
        },
        {
            "name": "rf_regularized_sqrt",
            "target_transform": "sqrt",
            "params": {
                "n_estimators": 700,
                "max_depth": 18,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": 0.70,
            },
        },
    ]


def make_stratification_bins(y: np.ndarray) -> Any:
    """Builds stratification bins for regression split."""

    try:
        bins = pd.qcut(np.log1p(y), q=5, duplicates="drop")

        if bins.isna().any():
            return None

        counts = pd.Series(bins).value_counts()

        if (counts < 2).any():
            return None

        return bins
    except Exception:
        return None


def select_best_candidate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    priors: dict[str, Any],
) -> tuple[Pipeline, dict[str, Any], dict[str, float]]:
    """Trains several Random Forest candidates and returns the best one."""

    train_features_df = apply_prior_features(train_df, priors)
    test_features_df = apply_prior_features(test_df, priors)

    X_train = train_features_df[ALL_FEATURES].copy()
    X_test = test_features_df[ALL_FEATURES].copy()

    y_train_raw = train_features_df[TARGET_COL].astype(float).to_numpy()
    y_test_raw = test_features_df[TARGET_COL].astype(float).to_numpy()

    weights = build_sample_weights(y_train_raw)

    best_model = None
    best_config = None
    best_metrics = None
    best_score = float("inf")

    for config in get_candidate_configs():
        method = config["target_transform"]

        y_train = transform_target(y_train_raw, method)

        pipeline = build_model_pipeline(config["params"])

        pipeline.fit(
            X_train,
            y_train,
            model__sample_weight=weights,
        )

        y_pred_transformed = pipeline.predict(X_test)
        y_pred_raw = inverse_transform_target(y_pred_transformed, method)
        y_pred_raw = np.clip(y_pred_raw, 0, None)

        metrics = evaluate_predictions(y_test_raw, y_pred_raw)

        selection_score = (
            metrics["wape_pct"]
            + metrics["smape_pct"] * 0.25
            + metrics["rmse"] * 0.03
            - metrics["r2"] * 5
        )

        log.info(
            "Candidate %s | score=%.4f | metrics=%s",
            config["name"],
            selection_score,
            json.dumps(metrics, ensure_ascii=False),
        )

        if selection_score < best_score:
            best_score = selection_score
            best_model = pipeline
            best_config = config
            best_metrics = metrics

    if best_model is None or best_config is None or best_metrics is None:
        raise RuntimeError("No model candidate could be trained.")

    return best_model, best_config, best_metrics


def train_single_model(dataset_df: pd.DataFrame, model_name: str) -> dict[str, Any]:
    """Trains one optimized model over a given dataset."""

    if len(dataset_df) < 10:
        raise ValueError(f"Not enough rows to train model {model_name}")

    dataset_df = dataset_df.copy().reset_index(drop=True)

    y = dataset_df[TARGET_COL].astype(float).to_numpy()
    stratify_bins = make_stratification_bins(y)

    test_size = 0.20 if len(dataset_df) >= 60 else 0.30

    train_df, test_df = train_test_split(
        dataset_df,
        test_size=test_size,
        random_state=42,
        stratify=stratify_bins,
    )

    priors_train = fit_prior_maps(train_df)

    _, best_config, validation_metrics = select_best_candidate(
        train_df=train_df,
        test_df=test_df,
        priors=priors_train,
    )

    log.info("Best config for %s: %s", model_name, best_config["name"])

    final_priors = fit_prior_maps(dataset_df)
    final_features_df = apply_prior_features(dataset_df, final_priors)

    X_full = final_features_df[ALL_FEATURES].copy()
    y_full_raw = final_features_df[TARGET_COL].astype(float).to_numpy()
    y_full = transform_target(y_full_raw, best_config["target_transform"])
    weights_full = build_sample_weights(y_full_raw)

    final_model = build_model_pipeline(best_config["params"])
    final_model.fit(
        X_full,
        y_full,
        model__sample_weight=weights_full,
    )

    train_pred_transformed = final_model.predict(X_full)
    train_pred_raw = inverse_transform_target(
        train_pred_transformed,
        best_config["target_transform"],
    )
    train_pred_raw = np.clip(train_pred_raw, 0, None)

    train_metrics = evaluate_predictions(y_full_raw, train_pred_raw)

    return {
        "model": final_model,
        "model_name": model_name,
        "model_type": "RandomForestRegressor",
        "best_config": best_config,
        "target_transform": best_config["target_transform"],
        "validation_metrics": validation_metrics,
        "train_metrics": train_metrics,
        "priors": final_priors,
        "training_rows": int(len(dataset_df)),
    }


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_random_forest_model() -> dict[str, Any]:
    """Trains optimized global and disease-specific Random Forest models."""

    log.info("Starting optimized Random Forest training")

    clean_df = load_clean_dataset()
    training_df = build_training_dataset(clean_df)

    missing_features = [
        column for column in [
            "year",
            "month",
            "disease",
            "department",
            "municipality",
            TARGET_COL,
        ]
        if column not in training_df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    log.info("Training global model with %s rows", f"{len(training_df):,}")
    global_payload = train_single_model(training_df, model_name="GLOBAL")

    disease_payloads: dict[str, Any] = {}

    for disease in sorted(training_df["disease"].dropna().unique()):
        disease_df = training_df[training_df["disease"] == disease].copy()

        if len(disease_df) < MIN_ROWS_FOR_DISEASE_MODEL:
            log.warning(
                "Skipping disease-specific model for %s. Only %s rows available.",
                disease,
                len(disease_df),
            )
            continue

        log.info(
            "Training disease-specific model for %s with %s rows",
            disease,
            f"{len(disease_df):,}",
        )

        disease_payloads[disease] = train_single_model(
            disease_df,
            model_name=disease,
        )

    metrics = {
        "GLOBAL": {
            "validation_metrics": global_payload["validation_metrics"],
            "train_metrics": global_payload["train_metrics"],
            "best_config": global_payload["best_config"]["name"],
            "training_rows": global_payload["training_rows"],
        },
        "BY_DISEASE": {
            disease: {
                "validation_metrics": payload["validation_metrics"],
                "train_metrics": payload["train_metrics"],
                "best_config": payload["best_config"]["name"],
                "training_rows": payload["training_rows"],
            }
            for disease, payload in disease_payloads.items()
        },
    }

    bundle = {
        "bundle_type": "optimized_random_forest_outbreak_bundle",
        "global_model": global_payload,
        "disease_models": disease_payloads,
        "target": TARGET_COL,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "all_features": ALL_FEATURES,
        "metrics": metrics,
        "climate_profiles": DEPARTMENT_CLIMATE_PROFILE,
        "default_climate_profile": DEFAULT_CLIMATE_PROFILE,
        "training_rows": int(len(training_df)),
    }

    joblib.dump(bundle, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log.info("Model bundle saved successfully: %s", MODEL_PATH)
    log.info("Metrics saved successfully: %s", METRICS_PATH)
    log.info("Final metrics:\n%s", json.dumps(metrics, ensure_ascii=False, indent=2))

    return bundle


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

def load_model_bundle() -> dict[str, Any]:
    """Loads model.pkl."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Train it first with: python -m src.ml.model --train"
        )

    return joblib.load(MODEL_PATH)


def select_payload_for_disease(bundle: dict[str, Any], disease: str) -> dict[str, Any]:
    """Selects disease-specific model if available; otherwise global model."""

    disease_norm = normalize_disease(disease)
    disease_models = bundle.get("disease_models", {})

    if disease_norm in disease_models:
        return disease_models[disease_norm]

    return bundle["global_model"]


def build_prediction_row(
    payload: dict[str, Any],
    month: int,
    department: str,
    city: str,
    disease: str = "Dengue",
) -> pd.DataFrame:
    """Builds one prediction row using user input."""

    if month < 1 or month > 12:
        raise ValueError("month must be between 1 and 12.")

    disease_norm = normalize_disease(disease)
    department_norm = normalize_name(department)
    municipality_norm = normalize_name(city)

    profile = get_climate_profile(department_norm)
    priors = payload["priors"]

    is_rainy = is_rainy_month(month)

    climate_risk_score = compute_climate_risk_score(
        avg_temp_c=float(profile["avg_temp_c"]),
        precipitation_mm=float(profile["precipitation_mm"]),
        humidity_pct=float(profile["humidity_pct"]),
        tropical_score=float(profile["tropical_score"]),
        is_rainy_season=is_rainy,
    )

    global_mean = float(priors.get("global_mean_cases", 0.0))

    municipality_fallback = lookup_value(
        priors,
        "municipality_global_mean_cases",
        make_key(department_norm, municipality_norm),
        global_mean,
    )

    previous_1 = get_previous_month_value(
        priors,
        disease_norm,
        department_norm,
        municipality_norm,
        month - 1,
        municipality_fallback,
    )

    previous_2 = get_previous_month_value(
        priors,
        disease_norm,
        department_norm,
        municipality_norm,
        month - 2,
        previous_1,
    )

    previous_3 = get_previous_month_value(
        priors,
        disease_norm,
        department_norm,
        municipality_norm,
        month - 3,
        previous_2,
    )

    row = {
        "disease": disease_norm,
        "year": 2024,
        "department": department_norm,
        "municipality": municipality_norm,
        "month": int(month),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "is_rainy_season": int(is_rainy),
        "thermal_floor": profile["thermal_floor"],
        "avg_temp_c": float(profile["avg_temp_c"]),
        "precipitation_mm": float(profile["precipitation_mm"]),
        "humidity_pct": float(profile["humidity_pct"]),
        "tropical_score": float(profile["tropical_score"]),
        "climate_risk_score": float(climate_risk_score),
        "is_tropical_climate": "SI" if float(profile["tropical_score"]) >= 0.70 else "NO",
        "is_vector_favorable": (
            "SI"
            if (
                float(profile["avg_temp_c"]) >= 22
                and float(profile["humidity_pct"]) >= 70
                and float(profile["tropical_score"]) >= 0.65
            )
            else "NO"
        ),
        "cases_previous_month": float(previous_1),
        "cases_previous_2_month_mean": float(np.mean([previous_1, previous_2])),
        "cases_previous_3_month_mean": float(np.mean([previous_1, previous_2, previous_3])),
        TARGET_COL: 0,
    }

    row_df = pd.DataFrame([row])
    row_df = apply_prior_features(row_df, priors)

    return row_df


def predict_outbreaks(
    month: int,
    department: str,
    city: str,
    disease: str = "Dengue",
) -> dict[str, Any]:
    """Predicts estimated cases/outbreak proxy for user input."""

    bundle = load_model_bundle()
    payload = select_payload_for_disease(bundle, disease)

    model: Pipeline = payload["model"]
    target_transform = payload["target_transform"]

    input_df = build_prediction_row(
        payload=payload,
        month=month,
        department=department,
        city=city,
        disease=disease,
    )

    prediction_transformed = model.predict(input_df[ALL_FEATURES])[0]
    prediction_cases = inverse_transform_target(
        np.array([prediction_transformed]),
        target_transform,
    )[0]

    prediction_cases = max(0.0, float(prediction_cases))

    estimated_cases = int(round(prediction_cases))
    outbreak_level = get_outbreak_level(prediction_cases)

    result = {
        "disease": normalize_disease(disease),
        "month": int(month),
        "department": normalize_name(department),
        "city": normalize_name(city),
        "estimated_cases": estimated_cases,
        "estimated_outbreak_proxy": estimated_cases,
        "outbreak_level": outbreak_level,
        "model_used": payload["model_name"],
        "thermal_floor": input_df.loc[0, "thermal_floor"],
        "avg_temp_c": round(float(input_df.loc[0, "avg_temp_c"]), 2),
        "precipitation_mm": round(float(input_df.loc[0, "precipitation_mm"]), 2),
        "humidity_pct": round(float(input_df.loc[0, "humidity_pct"]), 2),
        "tropical_score": round(float(input_df.loc[0, "tropical_score"]), 2),
        "climate_risk_score": round(float(input_df.loc[0, "climate_risk_score"]), 4),
        "is_rainy_season": int(input_df.loc[0, "is_rainy_season"]),
        "is_vector_favorable": input_df.loc[0, "is_vector_favorable"],
        "cases_previous_month": round(float(input_df.loc[0, "cases_previous_month"]), 2),
        "cases_previous_2_month_mean": round(float(input_df.loc[0, "cases_previous_2_month_mean"]), 2),
        "cases_previous_3_month_mean": round(float(input_df.loc[0, "cases_previous_3_month_mean"]), 2),
        "model_validation_metrics": payload.get("validation_metrics", {}),
        "model_train_metrics": payload.get("train_metrics", {}),
    }

    return result


# -----------------------------------------------------------------------------
# Batch prediction helper
# -----------------------------------------------------------------------------

def predict_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts multiple rows.

    Required input columns:
        month, department, city

    Optional:
        disease
    """

    results = []

    for _, row in input_df.iterrows():
        disease = row.get("disease", "Dengue")

        prediction = predict_outbreaks(
            month=int(row["month"]),
            department=str(row["department"]),
            city=str(row["city"]),
            disease=str(disease),
        )

        results.append(prediction)

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def run_training() -> None:
    """Runs model training."""

    log.info("=== Optimized Random Forest training pipeline started ===")

    bundle = train_random_forest_model()

    log.info("=== Training completed ===")
    log.info("Training rows: %s", bundle["training_rows"])
    log.info("Metrics: %s", json.dumps(bundle["metrics"], ensure_ascii=False))


def run_prediction(
    month: int,
    department: str,
    city: str,
    disease: str,
) -> None:
    """Runs one prediction from CLI."""

    prediction = predict_outbreaks(
        month=month,
        department=department,
        city=city,
        disease=disease,
    )

    print("\n── Predicción epidemiológica ─────────────────────────")
    print(json.dumps(prediction, ensure_ascii=False, indent=2))


def main() -> None:
    """Command line entrypoint."""

    parser = argparse.ArgumentParser(
        description="Optimized Random Forest model for mosquito-borne disease outbreak prediction.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train optimized Random Forest model and generate model.pkl.",
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run a prediction using an existing model.pkl.",
    )

    parser.add_argument(
        "--month",
        type=int,
        default=5,
        help="Prediction month. Example: 5",
    )

    parser.add_argument(
        "--department",
        type=str,
        default="NARIÑO",
        help="Department name. Example: NARIÑO",
    )

    parser.add_argument(
        "--city",
        type=str,
        default="PASTO",
        help="City or municipality name. Example: PASTO",
    )

    parser.add_argument(
        "--disease",
        type=str,
        default="Dengue",
        help="Disease name: Dengue, Chikungunya or Malaria.",
    )

    args = parser.parse_args()

    if args.train:
        run_training()
        return

    if args.predict:
        run_prediction(
            month=args.month,
            department=args.department,
            city=args.city,
            disease=args.disease,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()