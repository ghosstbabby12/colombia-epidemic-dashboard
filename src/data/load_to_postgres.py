"""
Carga los datasets procesados de SIVIGILA en PostgreSQL.

Tablas que crea:
  - casos_individuales  → Un registro por caso (311,930 filas)
  - casos_agregados     → Totales por enfermedad + departamento + semana (2,134 filas)

Uso:
  python src/data/load_to_postgres.py
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "dengue_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_engine():
    engine = create_engine(DATABASE_URL, echo=False)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    log.info("Conexión exitosa a PostgreSQL (%s:%s/%s)", DB_HOST, DB_PORT, DB_NAME)
    return engine


def cargar_casos_individuales(engine) -> None:
    path = PROCESSED_DIR / "sivigila_clean.parquet"
    if not path.exists():
        log.error("Archivo no encontrado: %s — ejecuta preprocessing.py primero", path)
        return

    log.info("Cargando casos individuales...")
    df = pd.read_parquet(path)

    # Columnas más útiles para consultas del dashboard y la API
    cols = [
        "CONSECUTIVE", "enfermedad", "Nombre_evento",
        "ANO", "SEMANA", "FEC_NOT", "INI_SIN",
        "EDAD", "edad_anos", "grupo_etario", "SEXO",
        "estrato", "regimen_salud", "tipo_caso",
        "hospitalizado", "condicion_final",
        "GP_GESTAN", "area_residencia",
        "COD_DPTO_O", "COD_MUN_O",
        "Departamento_ocurrencia", "Municipio_ocurrencia",
        "Departamento_residencia", "Municipio_residencia",
        "confirmados", "dias_notificacion",
    ]
    cols_exist = [c for c in cols if c in df.columns]
    df = df[cols_exist].copy()

    # Asegurar tipos compatibles con PostgreSQL
    df["grupo_etario"] = df["grupo_etario"].astype(str)
    for col in df.select_dtypes(include="Int64").columns:
        df[col] = df[col].astype("float64")  # Int64 → float para evitar errores

    log.info("Subiendo %d filas a tabla 'casos_individuales'...", len(df))
    df.to_sql(
        "casos_individuales",
        engine,
        if_exists="replace",
        index=False,
        chunksize=10_000,
        method="multi",
    )
    log.info("✓ casos_individuales cargada")


def cargar_casos_agregados(engine) -> None:
    path = PROCESSED_DIR / "sivigila_agg.parquet"
    if not path.exists():
        log.error("Archivo no encontrado: %s — ejecuta preprocessing.py primero", path)
        return

    log.info("Cargando casos agregados...")
    df = pd.read_parquet(path)

    for col in df.select_dtypes(include="Int64").columns:
        df[col] = df[col].astype("float64")

    log.info("Subiendo %d filas a tabla 'casos_agregados'...", len(df))
    df.to_sql(
        "casos_agregados",
        engine,
        if_exists="replace",
        index=False,
        chunksize=5_000,
        method="multi",
    )
    log.info("✓ casos_agregados cargada")


def crear_indices(engine) -> None:
    """Índices para acelerar las consultas del dashboard."""
    indices = [
        "CREATE INDEX IF NOT EXISTS idx_casos_ind_dpto   ON casos_individuales (\"Departamento_ocurrencia\")",
        "CREATE INDEX IF NOT EXISTS idx_casos_ind_semana ON casos_individuales (\"SEMANA\", \"ANO\")",
        "CREATE INDEX IF NOT EXISTS idx_casos_ind_enf    ON casos_individuales (enfermedad)",
        "CREATE INDEX IF NOT EXISTS idx_casos_agg_dpto   ON casos_agregados (departamento)",
        "CREATE INDEX IF NOT EXISTS idx_casos_agg_semana ON casos_agregados (\"SEMANA\", año)",
        "CREATE INDEX IF NOT EXISTS idx_casos_agg_enf    ON casos_agregados (enfermedad)",
    ]
    with engine.connect() as conn:
        for sql in indices:
            conn.execute(text(sql))
        conn.commit()
    log.info("✓ Índices creados")


def verificar_carga(engine) -> None:
    queries = {
        "casos_individuales": 'SELECT COUNT(*) FROM casos_individuales',
        "casos_agregados":    'SELECT COUNT(*) FROM casos_agregados',
        "casos por enfermedad": """
            SELECT enfermedad, COUNT(*) as casos
            FROM casos_individuales
            GROUP BY enfermedad
            ORDER BY casos DESC
        """,
        "top 5 dptos dengue": """
            SELECT "Departamento_ocurrencia", COUNT(*) as casos
            FROM casos_individuales
            WHERE enfermedad = 'Dengue'
            GROUP BY "Departamento_ocurrencia"
            ORDER BY casos DESC
            LIMIT 5
        """,
    }
    with engine.connect() as conn:
        for nombre, sql in queries.items():
            result = conn.execute(text(sql))
            log.info("── %s:\n%s", nombre, pd.DataFrame(result.fetchall()).to_string())


def run() -> None:
    log.info("=== Carga a PostgreSQL ===")
    engine = get_engine()
    cargar_casos_individuales(engine)
    cargar_casos_agregados(engine)
    crear_indices(engine)
    verificar_carga(engine)
    log.info("=== Carga completada ===")


if __name__ == "__main__":
    run()
