"""
API REST con FastAPI para el Dashboard Epidemiológico de Colombia.

Endpoints principales:
  GET  /health                           → Estado del servicio
  GET  /departments                      → Lista de departamentos disponibles
  GET  /cases/{department}               → Casos históricos por departamento
  GET  /risk/{department}                → Nivel de riesgo actual
  POST /predict                          → Predicción de riesgo dado un conjunto de features
  GET  /forecast/{department}            → Pronóstico de casos (próximas N semanas)

Ejecutar localmente:
  uvicorn src.backend.app:app --reload --port 8000
"""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "ml" / "models"

app = FastAPI(
    title="Colombia Epidemic Dashboard API",
    description="API para análisis y predicción de dengue en Colombia.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restringir en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Estado global (cargado una sola vez al iniciar) ───────────────────────────

_df: pd.DataFrame | None = None
_model = None
_le = None


def _load_artifacts():
    global _df, _model, _le

    features_path = PROCESSED_DIR / "features.parquet"
    if features_path.exists():
        _df = pd.read_parquet(features_path)
        log.info("Dataset cargado: %d filas", len(_df))
    else:
        log.warning("features.parquet no encontrado. Algunos endpoints retornarán vacíos.")

    try:
        import joblib
        _model = joblib.load(MODELS_DIR / "xgb_risk.joblib")
        _le = joblib.load(MODELS_DIR / "label_encoder.joblib")
        log.info("Modelos XGBoost cargados.")
    except FileNotFoundError:
        log.warning("Modelos no encontrados. Ejecuta src/ml/model.py primero.")


@app.on_event("startup")
async def startup_event():
    _load_artifacts()


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    week: int = Field(..., ge=1, le=52, description="Semana epidemiológica (1-52)")
    month: int = Field(..., ge=1, le=12)
    is_rainy_season: int = Field(0, ge=0, le=1)
    cases_lag_2w: float = Field(0.0, ge=0)
    cases_lag_4w: float = Field(0.0, ge=0)
    cases_lag_8w: float = Field(0.0, ge=0)
    cases_ma4: float = Field(0.0, ge=0)
    temp_avg: float = Field(27.0)
    precipitation_mm: float = Field(80.0, ge=0)
    humidity_pct: float = Field(70.0, ge=0, le=100)
    precip_x_humidity: float = Field(0.0, ge=0)


class PredictResponse(BaseModel):
    risk_level: Literal["bajo", "medio", "alto"]
    probability: float
    message: str


class ForecastPoint(BaseModel):
    date: str
    predicted_cases: float
    lower_bound: float
    upper_bound: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
def health_check():
    return {
        "status": "ok",
        "dataset_loaded": _df is not None,
        "model_loaded": _model is not None,
    }


@app.get("/departments", tags=["Datos"])
def get_departments():
    if _df is None:
        raise HTTPException(503, "Dataset no disponible.")
    depts = sorted(_df["department"].dropna().unique().tolist())
    return {"departments": depts, "total": len(depts)}


@app.get("/cases/{department}", tags=["Datos"])
def get_cases(department: str, year: int | None = None):
    if _df is None:
        raise HTTPException(503, "Dataset no disponible.")

    dept_upper = department.upper()
    mask = _df["department"] == dept_upper
    if year:
        mask &= _df["year"] == year

    subset = _df[mask][["year", "week", "department", "total_cases"]].copy()
    if subset.empty:
        raise HTTPException(404, f"No se encontraron datos para '{department}'.")

    return {
        "department": dept_upper,
        "year": year,
        "records": subset.to_dict(orient="records"),
    }


@app.get("/risk/{department}", tags=["Predicción"])
def get_current_risk(department: str):
    """Retorna el nivel de riesgo de la semana más reciente disponible."""
    if _df is None or _model is None:
        raise HTTPException(503, "Modelos o datos no disponibles.")

    dept_upper = department.upper()
    subset = _df[_df["department"] == dept_upper]
    if subset.empty:
        raise HTTPException(404, f"Departamento '{department}' no encontrado.")

    latest = subset.sort_values(["year", "week"]).iloc[[-1]]

    from src.ml.model import predict_risk
    result = predict_risk(_model, _le, latest)

    return {
        "department": dept_upper,
        "year": int(latest["year"].iloc[0]),
        "week": int(latest["week"].iloc[0]),
        "total_cases": int(latest["total_cases"].iloc[0]),
        "risk_level": result["predicted_risk"].iloc[0],
        "probability": float(result["risk_proba"].iloc[0]),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Predicción"])
def predict_risk_endpoint(body: PredictRequest):
    if _model is None or _le is None:
        raise HTTPException(503, "Modelo no disponible. Ejecuta src/ml/model.py primero.")

    import numpy as np

    features = [
        "week", "month", "is_rainy_season",
        "cases_lag_2w", "cases_lag_4w", "cases_lag_8w",
        "cases_ma4", "temp_avg", "precipitation_mm",
        "humidity_pct", "precip_x_humidity",
    ]
    X = pd.DataFrame([body.model_dump()])[features].astype(float)

    y_pred = _model.predict(X)[0]
    proba = float(_model.predict_proba(X).max())
    risk = _le.inverse_transform([y_pred])[0]

    messages = {
        "bajo": "Situación bajo control. Vigilancia rutinaria.",
        "medio": "Riesgo moderado. Reforzar acciones de prevención.",
        "alto": "ALERTA: Riesgo alto de brote. Activar protocolo de respuesta.",
    }

    return PredictResponse(
        risk_level=risk,
        probability=round(proba, 3),
        message=messages[risk],
    )


@app.get("/forecast/{department}", response_model=list[ForecastPoint], tags=["Predicción"])
def get_forecast(department: str, weeks: int = 8):
    if _df is None:
        raise HTTPException(503, "Dataset no disponible.")
    if weeks < 1 or weeks > 26:
        raise HTTPException(400, "El parámetro 'weeks' debe estar entre 1 y 26.")

    from src.ml.model import train_prophet_forecast

    forecast = train_prophet_forecast(_df, department=department, periods=weeks)
    if forecast.empty:
        raise HTTPException(
            404,
            f"No se pudo generar pronóstico para '{department}'. "
            "Verifica que haya suficientes datos históricos.",
        )

    return [
        ForecastPoint(
            date=str(row["ds"].date()),
            predicted_cases=round(max(row["yhat"], 0), 1),
            lower_bound=round(max(row["yhat_lower"], 0), 1),
            upper_bound=round(max(row["yhat_upper"], 0), 1),
        )
        for _, row in forecast.iterrows()
    ]
