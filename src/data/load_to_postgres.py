"""
Loads clean epidemiological datasets into local PostgreSQL.

Main responsibilities:
- Creates the target database if it does not exist.
- Creates public dashboard tables.
- Creates a simple DWH schema.
- Loads clean and aggregated data.
- Creates indexes for API/dashboard queries.

Usage:
    python src/data/load_to_postgres.py
    python -m src.data.load_to_postgres
"""

import logging
import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql
from sqlalchemy import create_engine, text


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
# Project paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# -----------------------------------------------------------------------------
# Database configuration
# -----------------------------------------------------------------------------

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "dengue_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")
LOAD_IF_EXISTS = os.getenv("LOAD_IF_EXISTS", "replace").lower().strip()

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
MAINTENANCE_DB = os.getenv("DB_MAINTENANCE_NAME", "postgres")


# -----------------------------------------------------------------------------
# Database creation
# -----------------------------------------------------------------------------

def create_database_if_not_exists() -> None:
    """
    Creates target PostgreSQL database if it does not exist.

    Comentario:
    PostgreSQL no permite crear una base de datos conectado a esa misma base.
    Por eso primero se conecta a la base de mantenimiento 'postgres'.
    """

    log.info("Checking if database exists: %s", DB_NAME)

    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=MAINTENANCE_DB,
        user=DB_USER,
        password=DB_PASS,
    )

    connection.autocommit = True

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (DB_NAME,),
            )

            exists = cursor.fetchone() is not None

            if exists:
                log.info("Database already exists: %s", DB_NAME)
                return

            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)),
            )

            log.info("Database created successfully: %s", DB_NAME)

    finally:
        connection.close()


def get_engine():
    """Returns SQLAlchemy engine connected to target database."""

    engine = create_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    log.info("Connected to PostgreSQL: %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)

    return engine


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------

def read_processed_dataset(base_name: str) -> pd.DataFrame:
    """Reads processed dataset from Parquet or CSV."""

    parquet_path = PROCESSED_DIR / f"{base_name}.parquet"
    csv_path = PROCESSED_DIR / f"{base_name}.csv"

    if parquet_path.exists():
        log.info("Reading processed Parquet: %s", parquet_path.name)
        return pd.read_parquet(parquet_path)

    if csv_path.exists():
        log.info("Reading processed CSV: %s", csv_path.name)
        return pd.read_csv(csv_path, dtype=str)

    raise FileNotFoundError(
        f"Processed dataset not found: {parquet_path.name} or {csv_path.name}",
    )


def prepare_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    """Converts pandas-specific values to SQL-friendly values."""

    df = df.copy()

    for column in df.columns:
        if str(df[column].dtype) == "category":
            df[column] = df[column].astype(str)

        if str(df[column].dtype).startswith("string"):
            df[column] = df[column].astype(object)

    df = df.replace({pd.NA: None})
    df = df.where(pd.notnull(df), None)

    return df


# -----------------------------------------------------------------------------
# Dashboard tables
# -----------------------------------------------------------------------------

def load_dashboard_tables(engine) -> None:
    """
    Loads tables used directly by the FastAPI dashboard.

    Tables:
    - public.casos_individuales
    - public.casos_agregados
    """

    clean_df = read_processed_dataset("sivigila_clean")
    agg_df = read_processed_dataset("sivigila_agg")

    clean_df = prepare_for_sql(clean_df)
    agg_df = prepare_for_sql(agg_df)

    log.info("Loading public.casos_individuales with %s rows", f"{len(clean_df):,}")

    clean_df.to_sql(
        "casos_individuales",
        engine,
        schema="public",
        if_exists=LOAD_IF_EXISTS,
        index=False,
        chunksize=10_000,
        method="multi",
    )

    log.info("Loading public.casos_agregados with %s rows", f"{len(agg_df):,}")

    agg_df.to_sql(
        "casos_agregados",
        engine,
        schema="public",
        if_exists=LOAD_IF_EXISTS,
        index=False,
        chunksize=5_000,
        method="multi",
    )

    log.info("Dashboard tables loaded successfully")


# -----------------------------------------------------------------------------
# DWH schema
# -----------------------------------------------------------------------------

def create_dwh_schema(engine) -> None:
    """Creates DWH schema if it does not exist."""

    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS dwh"))

    log.info("DWH schema ready")


def build_dwh_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds dimensional model from processed data.

    DWH model:
    - dwh.dim_enfermedad
    - dwh.dim_geografia
    - dwh.dim_tiempo
    - dwh.fact_casos_semanales
    """

    clean_df = read_processed_dataset("sivigila_clean")
    agg_df = read_processed_dataset("sivigila_agg")

    disease_codes = (
        clean_df[["enfermedad", "COD_EVE"]]
        .dropna(subset=["enfermedad"])
        .drop_duplicates()
        .rename(columns={"COD_EVE": "codigo_evento"})
    )

    dim_disease = (
        disease_codes
        .sort_values("enfermedad")
        .reset_index(drop=True)
    )

    dim_disease.insert(0, "disease_id", range(1, len(dim_disease) + 1))

    dim_geo = (
        agg_df[["cod_dpto", "departamento"]]
        .drop_duplicates()
        .sort_values(["departamento"])
        .reset_index(drop=True)
    )

    dim_geo["municipio"] = "TODOS"
    dim_geo["cod_mun"] = None

    dim_geo.insert(0, "geo_id", range(1, len(dim_geo) + 1))

    dim_time = (
        agg_df[["año", "SEMANA", "month"]]
        .drop_duplicates()
        .sort_values(["año", "SEMANA"])
        .reset_index(drop=True)
    )

    dim_time["periodo"] = (
        dim_time["año"].astype(str)
        + "-W"
        + dim_time["SEMANA"].astype(str).str.zfill(2)
    )

    dim_time.insert(0, "time_id", range(1, len(dim_time) + 1))

    fact_df = agg_df.copy()

    fact_df = fact_df.merge(
        dim_disease[["disease_id", "enfermedad"]],
        on="enfermedad",
        how="left",
    )

    fact_df = fact_df.merge(
        dim_geo[["geo_id", "cod_dpto", "departamento"]],
        on=["cod_dpto", "departamento"],
        how="left",
    )

    fact_df = fact_df.merge(
        dim_time[["time_id", "año", "SEMANA"]],
        on=["año", "SEMANA"],
        how="left",
    )

    fact_columns = [
        "disease_id",
        "geo_id",
        "time_id",
        "total_casos",
        "confirmados",
        "hospitalizados",
        "fallecidos",
        "edad_promedio",
        "pct_femenino",
        "risk_label",
    ]

    fact_df = fact_df[fact_columns].copy()
    fact_df.insert(0, "fact_id", range(1, len(fact_df) + 1))

    return dim_disease, dim_geo, dim_time, fact_df


def load_dwh_tables(engine) -> None:
    """Loads dimensional warehouse tables."""

    create_dwh_schema(engine)

    dim_disease, dim_geo, dim_time, fact_df = build_dwh_tables()

    tables = {
        "dim_enfermedad": dim_disease,
        "dim_geografia": dim_geo,
        "dim_tiempo": dim_time,
        "fact_casos_semanales": fact_df,
    }

    for table_name, df in tables.items():
        df = prepare_for_sql(df)

        log.info("Loading dwh.%s with %s rows", table_name, f"{len(df):,}")

        df.to_sql(
            table_name,
            engine,
            schema="dwh",
            if_exists=LOAD_IF_EXISTS,
            index=False,
            chunksize=5_000,
            method="multi",
        )

    log.info("DWH tables loaded successfully")


# -----------------------------------------------------------------------------
# Indexes and validation
# -----------------------------------------------------------------------------

def create_indexes(engine) -> None:
    """Creates indexes to speed up dashboard queries."""

    statements = [
        'CREATE INDEX IF NOT EXISTS idx_casos_ind_enfermedad ON public.casos_individuales (enfermedad)',
        'CREATE INDEX IF NOT EXISTS idx_casos_ind_semana ON public.casos_individuales ("SEMANA", "ANO")',
        'CREATE INDEX IF NOT EXISTS idx_casos_ind_dpto ON public.casos_individuales ("Departamento_ocurrencia")',
        'CREATE INDEX IF NOT EXISTS idx_casos_ind_mun ON public.casos_individuales ("Municipio_ocurrencia")',
        'CREATE INDEX IF NOT EXISTS idx_casos_agg_enfermedad ON public.casos_agregados (enfermedad)',
        'CREATE INDEX IF NOT EXISTS idx_casos_agg_semana ON public.casos_agregados ("SEMANA", año)',
        'CREATE INDEX IF NOT EXISTS idx_casos_agg_dpto ON public.casos_agregados (departamento)',
        'CREATE INDEX IF NOT EXISTS idx_dim_enfermedad_id ON dwh.dim_enfermedad (disease_id)',
        'CREATE INDEX IF NOT EXISTS idx_dim_geo_id ON dwh.dim_geografia (geo_id)',
        'CREATE INDEX IF NOT EXISTS idx_dim_tiempo_id ON dwh.dim_tiempo (time_id)',
        'CREATE INDEX IF NOT EXISTS idx_fact_disease ON dwh.fact_casos_semanales (disease_id)',
        'CREATE INDEX IF NOT EXISTS idx_fact_geo ON dwh.fact_casos_semanales (geo_id)',
        'CREATE INDEX IF NOT EXISTS idx_fact_time ON dwh.fact_casos_semanales (time_id)',
    ]

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))

    log.info("Indexes created successfully")


def verify_load(engine) -> None:
    """Prints verification queries after load."""

    queries = {
        "public.casos_individuales": "SELECT COUNT(*) AS total FROM public.casos_individuales",
        "public.casos_agregados": "SELECT COUNT(*) AS total FROM public.casos_agregados",
        "dwh.dim_enfermedad": "SELECT COUNT(*) AS total FROM dwh.dim_enfermedad",
        "dwh.dim_geografia": "SELECT COUNT(*) AS total FROM dwh.dim_geografia",
        "dwh.dim_tiempo": "SELECT COUNT(*) AS total FROM dwh.dim_tiempo",
        "dwh.fact_casos_semanales": "SELECT COUNT(*) AS total FROM dwh.fact_casos_semanales",
        "cases_by_disease": """
            SELECT enfermedad, COUNT(*) AS total
            FROM public.casos_individuales
            GROUP BY enfermedad
            ORDER BY total DESC
        """,
    }

    with engine.connect() as connection:
        for name, query in queries.items():
            result = connection.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

            log.info("Verification - %s:\n%s", name, df.to_string(index=False))


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def run() -> None:
    """Runs the complete PostgreSQL load."""

    log.info("Starting PostgreSQL load process")

    create_database_if_not_exists()

    engine = get_engine()

    load_dashboard_tables(engine)
    load_dwh_tables(engine)
    create_indexes(engine)
    verify_load(engine)

    log.info("PostgreSQL load completed successfully")


if __name__ == "__main__":
    run()