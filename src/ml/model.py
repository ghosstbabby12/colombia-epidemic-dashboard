"""
Módulo de Machine Learning para predicción de dengue.

Contiene:
  - Entrenamiento de modelo de clasificación de riesgo (XGBoost)
  - Predicción de series temporales por departamento (Prophet)
  - Serialización y carga de modelos
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[0] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "week", "month", "is_rainy_season",
    "cases_lag_2w", "cases_lag_4w", "cases_lag_8w",
    "cases_ma4",
    "temp_avg", "precipitation_mm", "humidity_pct",
    "precip_x_humidity",
]
TARGET_COL = "risk_label"


# ── Utilidades ────────────────────────────────────────────────────────────────

def load_features(path: Path | None = None) -> pd.DataFrame:
    path = path or PROCESSED_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {path}. Ejecuta preprocessing.py primero."
        )
    return pd.read_parquet(path)


def _available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


# ── Clasificación de riesgo con XGBoost ───────────────────────────────────────

def train_risk_classifier(df: pd.DataFrame) -> tuple[XGBClassifier, LabelEncoder]:
    """
    Entrena un clasificador XGBoost que predice el nivel de riesgo
    (bajo / medio / alto) para una semana y departamento dados.

    Retorna el modelo entrenado y el LabelEncoder de la variable target.
    """
    features = _available_features(df)
    X = df[features].fillna(0).astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    log.info("=== Reporte de clasificación ===")
    log.info(
        "\n%s",
        classification_report(y_test, y_pred, target_names=le.classes_),
    )

    joblib.dump(model, MODELS_DIR / "xgb_risk.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    log.info("Modelos guardados en %s", MODELS_DIR)

    return model, le


def load_risk_classifier() -> tuple[XGBClassifier, LabelEncoder]:
    model = joblib.load(MODELS_DIR / "xgb_risk.joblib")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    return model, le


def predict_risk(
    model: XGBClassifier,
    le: LabelEncoder,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Agrega columna 'predicted_risk' al dataframe de entrada."""
    features = _available_features(df)
    X = df[features].fillna(0).astype(float)
    y_encoded = model.predict(X)
    df = df.copy()
    df["predicted_risk"] = le.inverse_transform(y_encoded)
    df["risk_proba"] = model.predict_proba(X).max(axis=1).round(3)
    return df


# ── Predicción de series temporales con Prophet ───────────────────────────────

def train_prophet_forecast(
    df: pd.DataFrame,
    department: str,
    periods: int = 8,
) -> pd.DataFrame:
    """
    Ajusta un modelo Prophet para el departamento indicado y genera
    `periods` semanas de pronóstico.

    Retorna un DataFrame con columnas: ds, yhat, yhat_lower, yhat_upper.
    """
    try:
        from prophet import Prophet
    except ImportError:
        log.error("Prophet no está instalado. Ejecuta: pip install prophet")
        return pd.DataFrame()

    df_dept = (
        df[df["department"] == department.upper()]
        .sort_values(["year", "week"])
        .copy()
    )
    if len(df_dept) < 10:
        log.warning("Datos insuficientes para %s (%d filas).", department, len(df_dept))
        return pd.DataFrame()

    # Prophet requiere columnas 'ds' (fecha) y 'y' (valor)
    df_dept["ds"] = pd.to_datetime(
        df_dept["year"].astype(str) + "-W" + df_dept["week"].astype(str).str.zfill(2) + "-1",
        format="%Y-W%W-%w",
        errors="coerce",
    )
    df_prophet = df_dept[["ds", "total_cases"]].rename(columns={"total_cases": "y"}).dropna()

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=periods, freq="W")
    forecast = m.predict(future)

    mae = mean_absolute_error(
        df_prophet["y"],
        forecast.loc[forecast["ds"].isin(df_prophet["ds"]), "yhat"],
    )
    log.info("Prophet MAE (%s): %.2f casos/semana", department, mae)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)


# ── Pipeline principal ────────────────────────────────────────────────────────

def run() -> None:
    log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.info("=== Entrenamiento del modelo ===")

    df = load_features()
    log.info("Dataset cargado: %d filas", len(df))

    model, le = train_risk_classifier(df)
    log.info("Clasificador XGBoost entrenado y guardado.")

    dept_ejemplo = "ANTIOQUIA"
    forecast = train_prophet_forecast(df, department=dept_ejemplo, periods=8)
    if not forecast.empty:
        log.info("Pronóstico para %s (próximas 8 semanas):\n%s", dept_ejemplo, forecast.to_string())

    log.info("=== Pipeline ML completado ===")


if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
