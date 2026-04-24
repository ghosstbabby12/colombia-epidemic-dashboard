"""
Strict ETL preprocessing for SIVIGILA mosquito-borne disease datasets.

Inputs:
    data/raw/Datos_2024_210.xlsx
    data/raw/Datos_2024_217.xlsx
    data/raw/Datos_2024_490.xlsx
    data/raw/sivigila_dengue_medata_2024.csv
    data/raw/sivigila_chikungunya_medata_2024.csv

Outputs:
    data/processed/sivigila_clean.csv
    data/processed/sivigila_clean.parquet
    data/processed/sivigila_agg.csv
    data/processed/sivigila_agg.parquet
    data/processed/sivigila_rejected_rows.csv
    data/processed/quality_report.csv
    data/processed/quality_report.json
"""

import json
import logging
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------

SOURCE_FILES = [
    {
        "disease": "Dengue",
        "event_code": 210,
        "year": 2024,
        "source": "ins_sivigila",
        "path": RAW_DIR / "Datos_2024_210.xlsx",
    },
    {
        "disease": "Dengue",
        "event_code": 210,
        "year": 2024,
        "source": "medata",
        "path": RAW_DIR / "sivigila_dengue_medata_2024.csv",
    },
    {
        "disease": "Chikungunya",
        "event_code": 217,
        "year": 2024,
        "source": "ins_sivigila",
        "path": RAW_DIR / "Datos_2024_217.xlsx",
    },
    {
        "disease": "Chikungunya",
        "event_code": 217,
        "year": 2024,
        "source": "medata",
        "path": RAW_DIR / "sivigila_chikungunya_medata_2024.csv",
    },
    {
        "disease": "Malaria",
        "event_code": 490,
        "year": 2024,
        "source": "ins_sivigila",
        "path": RAW_DIR / "Datos_2024_490.xlsx",
    },
]


# -----------------------------------------------------------------------------
# Canonical columns and aliases
# -----------------------------------------------------------------------------

CANONICAL_COLUMNS = [
    "CONSECUTIVE",
    "COD_EVE",
    "Nombre_evento",
    "FEC_NOT",
    "SEMANA",
    "ANO",
    "INI_SIN",
    "EDAD",
    "UNI_MED",
    "SEXO",
    "estrato",
    "TIP_SS",
    "TIP_CAS",
    "PAC_HOS",
    "CON_FIN",
    "GP_GESTAN",
    "AREA",
    "COD_DPTO_O",
    "COD_MUN_O",
    "Departamento_ocurrencia",
    "Municipio_ocurrencia",
    "Departamento_residencia",
    "Municipio_residencia",
    "nom_est_f_caso",
    "confirmados",
]

COLUMN_ALIASES = {
    "CONSECUTIVE": [
        "CONSECUTIVE",
        "CONSECUTIVO",
        "ID",
        "ID_CASO",
        "CODIGO_CASO",
        "consecutive",
        "consecutivo",
    ],
    "COD_EVE": [
        "COD_EVE",
        "COD_EVENTO",
        "CODIGO_EVENTO",
        "codigo_evento",
        "cod_eve",
        "evento",
    ],
    "Nombre_evento": [
        "Nombre_evento",
        "NOMBRE_EVENTO",
        "NOM_EVE",
        "NOM_EVENTO",
        "evento_nombre",
        "nombre_evento",
    ],
    "FEC_NOT": [
        "FEC_NOT",
        "FECHA_NOTIFICACION",
        "FECHA DE NOTIFICACION",
        "fec_not",
        "fecha_notificacion",
    ],
    "SEMANA": [
        "SEMANA",
        "SEMANA_EPIDEMIOLOGICA",
        "SEM_EPI",
        "semana",
    ],
    "ANO": [
        "ANO",
        "AÑO",
        "ANIO",
        "YEAR",
        "ano",
        "anio",
        "año",
    ],
    "INI_SIN": [
        "INI_SIN",
        "FECHA_INICIO_SINTOMAS",
        "INICIO_SINTOMAS",
        "ini_sin",
        "fecha_inicio_sintomas",
    ],
    "EDAD": [
        "EDAD",
        "edad",
    ],
    "UNI_MED": [
        "UNI_MED",
        "UNIDAD_MEDIDA",
        "UNIDAD_EDAD",
        "uni_med",
    ],
    "SEXO": [
        "SEXO",
        "sexo",
        "GENERO",
        "genero",
    ],
    "estrato": [
        "estrato",
        "ESTRATO",
        "ESTRATO_SOCIOECONOMICO",
    ],
    "TIP_SS": [
        "TIP_SS",
        "TIPO_SS",
        "SEGURIDAD_SOCIAL",
        "REGIMEN",
        "tip_ss",
        "regimen",
    ],
    "TIP_CAS": [
        "TIP_CAS",
        "TIPO_CASO",
        "CLASIFICACION_CASO",
        "tip_cas",
    ],
    "PAC_HOS": [
        "PAC_HOS",
        "HOSPITALIZADO",
        "PACIENTE_HOSPITALIZADO",
        "pac_hos",
        "hospitalizado",
    ],
    "CON_FIN": [
        "CON_FIN",
        "CONDICION_FINAL",
        "condicion_final",
    ],
    "GP_GESTAN": [
        "GP_GESTAN",
        "GESTANTE",
        "gestante",
    ],
    "AREA": [
        "AREA",
        "AREA_RESIDENCIA",
        "area",
    ],
    "COD_DPTO_O": [
        "COD_DPTO_O",
        "COD_DEPTO_O",
        "COD_DPTO_OCURRENCIA",
        "CODIGO_DEPARTAMENTO_OCURRENCIA",
        "cod_dpto_o",
    ],
    "COD_MUN_O": [
        "COD_MUN_O",
        "COD_MPIO_O",
        "COD_MUNICIPIO_OCURRENCIA",
        "CODIGO_MUNICIPIO_OCURRENCIA",
        "cod_mun_o",
    ],
    "Departamento_ocurrencia": [
        "Departamento_ocurrencia",
        "DEPARTAMENTO_OCURRENCIA",
        "DEPTO_OCURRENCIA",
        "NOM_DPTO_O",
        "DEPARTAMENTO",
        "departamento_ocurrencia",
        "departamento",
    ],
    "Municipio_ocurrencia": [
        "Municipio_ocurrencia",
        "MUNICIPIO_OCURRENCIA",
        "NOM_MUN_O",
        "MUNICIPIO",
        "municipio_ocurrencia",
        "municipio",
    ],
    "Departamento_residencia": [
        "Departamento_residencia",
        "DEPARTAMENTO_RESIDENCIA",
        "NOM_DPTO_R",
        "departamento_residencia",
    ],
    "Municipio_residencia": [
        "Municipio_residencia",
        "MUNICIPIO_RESIDENCIA",
        "NOM_MUN_R",
        "municipio_residencia",
    ],
    "nom_est_f_caso": [
        "nom_est_f_caso",
        "NOM_EST_F_CASO",
        "ESTADO_FINAL_CASO",
        "estado_final_caso",
    ],
    "confirmados": [
        "confirmados",
        "CONFIRMADOS",
        "CONFIRMADO",
        "confirmado",
    ],
}


# -----------------------------------------------------------------------------
# Decoding maps
# -----------------------------------------------------------------------------

TIP_CAS_MAP = {
    1: "Sospechoso",
    2: "Probable",
    3: "Confirmado_Lab",
    4: "Confirmado_Clinico",
}

CON_FIN_MAP = {
    1: "Vivo",
    2: "Muerto",
}

PAC_HOS_MAP = {
    1: "Hospitalizado",
    2: "No hospitalizado",
}

AREA_MAP = {
    1: "Cabecera municipal",
    2: "Rural disperso",
    3: "Centro poblado",
}

UNI_MED_MAP = {
    1: "Años",
    2: "Meses",
    3: "Días",
    4: "Horas",
}

TIP_SS_MAP = {
    "C": "Contributivo",
    "S": "Subsidiado",
    "E": "Excepcion",
    "P": "Especial",
    "I": "Indeterminado",
    "N": "No asegurado",
}

MISSING_TEXT_VALUES = {
    "",
    "NA",
    "N/A",
    "NAN",
    "NONE",
    "NULL",
    "SIN DATO",
    "SIN_DATO",
    "NO APLICA",
    "NO APLICA.",
    "SD",
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def normalize_header(value: Any) -> str:
    """Normalizes column names to compare different formats."""

    text = str(value).strip()
    text = text.replace("\ufeff", "")

    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))

    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text


def normalize_text(value: Any) -> Any:
    """Normalizes text values without losing meaningful labels."""

    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if text.upper() in MISSING_TEXT_VALUES:
        return pd.NA

    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))

    text = re.sub(r"\s+", " ", text).strip().upper()

    if text in MISSING_TEXT_VALUES:
        return pd.NA

    return text


def to_nullable_int(series: pd.Series) -> pd.Series:
    """Converts a series to nullable integer."""

    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def to_nullable_float(series: pd.Series) -> pd.Series:
    """Converts a series to nullable float."""

    return pd.to_numeric(series, errors="coerce").astype("float64")


def normalize_code(value: Any, width: int) -> Any:
    """Normalizes DANE codes as zero-padded strings."""

    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if text.upper() in MISSING_TEXT_VALUES:
        return pd.NA

    text = re.sub(r"\.0$", "", text)
    digits = re.sub(r"\D", "", text)

    if not digits:
        return pd.NA

    return digits.zfill(width)


def iso_week_to_month(year: Any, week: Any) -> Any:
    """Returns month from ISO year/week."""

    try:
        return date.fromisocalendar(int(year), int(week), 1).month
    except Exception:
        return pd.NA


def get_risk_label(total_cases: int) -> str:
    """Classifies epidemiological risk based on weekly cases."""

    if total_cases <= 50:
        return "bajo"

    if total_cases <= 200:
        return "medio"

    return "alto"


def parse_date_series(series: pd.Series) -> pd.Series:
    """
    Parses date columns safely.

    Comentario:
    Primero intenta leer fechas en formato automático/mixed.
    Si alguna fecha queda vacía, intenta una segunda lectura con dayfirst=True.
    Esto evita warnings cuando las fechas vienen en formato YYYY-MM-DD.
    """

    text = series.astype("string").str.strip()

    parsed = pd.to_datetime(
        text,
        errors="coerce",
        format="mixed",
        dayfirst=False,
    )

    missing_mask = parsed.isna() & text.notna()

    if missing_mask.any():
        parsed.loc[missing_mask] = pd.to_datetime(
            text.loc[missing_mask],
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )

    return parsed


# -----------------------------------------------------------------------------
# Extraction helpers
# -----------------------------------------------------------------------------

def read_raw_file(path: Path) -> pd.DataFrame:
    """Reads CSV or XLSX files with robust fallbacks."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    log.info("Reading raw file: %s", path.name)

    if path.suffix.lower() == ".csv":
        encodings = ["utf-8-sig", "utf-8", "latin1"]

        last_error = None

        for encoding in encodings:
            try:
                return pd.read_csv(
                    path,
                    dtype=str,
                    sep=None,
                    engine="python",
                    encoding=encoding,
                    on_bad_lines="skip",
                )
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Could not read CSV file {path}: {last_error}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, engine="openpyxl")

    raise ValueError(f"Unsupported file format: {path.suffix}")


def collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses duplicated columns after alias renaming.

    Comentario:
    Si dos columnas originales caen en el mismo nombre canónico, se conserva
    el primer valor no nulo por fila.
    """

    duplicated_columns = df.columns[df.columns.duplicated()].unique()

    if len(duplicated_columns) == 0:
        return df

    for column in duplicated_columns:
        same_name_df = df.loc[:, df.columns == column]
        collapsed = same_name_df.bfill(axis=1).iloc[:, 0]

        df = df.drop(columns=[column])
        df[column] = collapsed

    return df


def standardize_columns(
    df: pd.DataFrame,
    disease: str,
    event_code: int,
    year: int,
    source: str,
    filename: str,
) -> pd.DataFrame:
    """Maps different source columns to a common SIVIGILA schema."""

    original_columns = list(df.columns)

    normalized_to_original = {
        normalize_header(column): column
        for column in original_columns
    }

    rename_map = {}

    for canonical_column, aliases in COLUMN_ALIASES.items():
        alias_keys = {normalize_header(alias) for alias in aliases}

        for normalized_column, original_column in normalized_to_original.items():
            if normalized_column in alias_keys:
                rename_map[original_column] = canonical_column
                break

    df = df.rename(columns=rename_map)
    df = collapse_duplicate_columns(df)

    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[CANONICAL_COLUMNS].copy()

    df["enfermedad"] = disease
    df["source_name"] = source
    df["source_file"] = filename

    if df["COD_EVE"].isna().all():
        df["COD_EVE"] = event_code

    if df["ANO"].isna().all():
        df["ANO"] = year

    if df["Nombre_evento"].isna().all():
        df["Nombre_evento"] = disease

    return df


# -----------------------------------------------------------------------------
# Cleaning functions
# -----------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies strict normalization, type casting and quality rules."""

    # Fechas
    for column in ["FEC_NOT", "INI_SIN"]:
        df[column] = parse_date_series(df[column])

    # Números enteros
    integer_columns = [
        "COD_EVE",
        "SEMANA",
        "ANO",
        "EDAD",
        "UNI_MED",
        "TIP_CAS",
        "PAC_HOS",
        "CON_FIN",
        "GP_GESTAN",
        "AREA",
        "confirmados",
    ]

    for column in integer_columns:
        df[column] = to_nullable_int(df[column])

    # Códigos DANE
    df["COD_DPTO_O"] = df["COD_DPTO_O"].apply(lambda value: normalize_code(value, 2))
    df["COD_MUN_O"] = df["COD_MUN_O"].apply(lambda value: normalize_code(value, 5))

    # Texto
    text_columns = [
        "CONSECUTIVE",
        "Nombre_evento",
        "SEXO",
        "TIP_SS",
        "Departamento_ocurrencia",
        "Municipio_ocurrencia",
        "Departamento_residencia",
        "Municipio_residencia",
        "nom_est_f_caso",
        "enfermedad",
        "source_name",
        "source_file",
    ]

    for column in text_columns:
        df[column] = df[column].apply(normalize_text)

    # Sexo
    df["SEXO"] = df["SEXO"].replace({
        "FEMENINO": "F",
        "MASCULINO": "M",
        "HOMBRE": "M",
        "MUJER": "F",
    })

    df.loc[~df["SEXO"].isin(["M", "F"]), "SEXO"] = pd.NA

    # Estrato
    df["estrato"] = to_nullable_int(df["estrato"])
    df.loc[~df["estrato"].between(0, 6), "estrato"] = pd.NA

    # Semana epidemiológica
    df.loc[~df["SEMANA"].between(1, 53), "SEMANA"] = pd.NA

    # Año
    df.loc[~df["ANO"].between(2000, 2100), "ANO"] = pd.NA

    # Edad normalizada en años
    # Comentario:
    # Se convierte a float64 para permitir decimales cuando la edad viene en meses,
    # días u horas. Esto corrige el error:
    # TypeError: cannot safely cast non-equivalent object to int64
    edad = pd.to_numeric(df["EDAD"], errors="coerce").astype("float64")
    unidad = pd.to_numeric(df["UNI_MED"], errors="coerce").astype("float64")

    edad_anos = edad.copy()

    mask_meses = unidad.eq(2)
    mask_dias = unidad.eq(3)
    mask_horas = unidad.eq(4)

    edad_anos.loc[mask_meses] = edad.loc[mask_meses] / 12.0
    edad_anos.loc[mask_dias] = edad.loc[mask_dias] / 365.0
    edad_anos.loc[mask_horas] = edad.loc[mask_horas] / 8760.0

    edad_anos = edad_anos.round(2)
    edad_anos.loc[(edad_anos < 0) | (edad_anos > 120)] = np.nan

    df["edad_anos"] = edad_anos

    # Grupo etario
    df["grupo_etario"] = pd.cut(
        df["edad_anos"],
        bins=[-1, 4, 14, 29, 44, 59, np.inf],
        labels=["0-4", "5-14", "15-29", "30-44", "45-59", "60+"],
    ).astype("string")

    df["grupo_etario"] = df["grupo_etario"].fillna("Sin dato")

    # Decodificaciones
    df["tipo_caso"] = df["TIP_CAS"].map(TIP_CAS_MAP).fillna("Sin dato")
    df["condicion_final"] = df["CON_FIN"].map(CON_FIN_MAP).fillna("Sin dato")
    df["hospitalizado"] = df["PAC_HOS"].map(PAC_HOS_MAP).fillna("Sin dato")
    df["area_residencia"] = df["AREA"].map(AREA_MAP).fillna("Sin dato")
    df["unidad_edad"] = df["UNI_MED"].map(UNI_MED_MAP).fillna("Sin dato")

    df["regimen_salud"] = df["TIP_SS"].map(TIP_SS_MAP)
    df["regimen_salud"] = df["regimen_salud"].fillna(df["TIP_SS"]).fillna("Sin dato")

    # Confirmados
    if df["confirmados"].isna().all():
        df["confirmados"] = 0

    df["confirmados"] = df["confirmados"].fillna(0).astype("Int64")

    confirmed_by_status = (
        df["nom_est_f_caso"]
        .fillna("")
        .astype(str)
        .str.contains("CONFIRMADO", case=False, na=False)
    )

    confirmed_by_case_type = df["TIP_CAS"].isin([3, 4])

    df.loc[confirmed_by_status | confirmed_by_case_type, "confirmados"] = 1

    # Tiempo de notificación
    df["dias_notificacion"] = (df["FEC_NOT"] - df["INI_SIN"]).dt.days
    df.loc[
        (df["dias_notificacion"] < 0) | (df["dias_notificacion"] > 365),
        "dias_notificacion",
    ] = np.nan

    # Campos categóricos mínimos
    fill_text_columns = [
        "Departamento_ocurrencia",
        "Municipio_ocurrencia",
        "Departamento_residencia",
        "Municipio_residencia",
        "SEXO",
    ]

    for column in fill_text_columns:
        df[column] = df[column].fillna("SIN_DATO")

    return df


def build_quality_report(df: pd.DataFrame, rejected_df: pd.DataFrame) -> pd.DataFrame:
    """Builds a quality report for auditing the ETL."""

    rows = []

    rows.append({
        "metric": "total_clean_rows",
        "value": len(df),
    })

    rows.append({
        "metric": "total_rejected_rows",
        "value": len(rejected_df),
    })

    for column in df.columns:
        rows.append({
            "metric": f"nulls_{column}",
            "value": int(df[column].isna().sum()),
        })

    for disease, count in df["enfermedad"].value_counts(dropna=False).items():
        rows.append({
            "metric": f"rows_disease_{disease}",
            "value": int(count),
        })

    return pd.DataFrame(rows)


def deduplicate_cases(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Removes duplicate case rows."""

    before = len(df)

    if "CONSECUTIVE" in df.columns:
        subset_columns = ["enfermedad", "CONSECUTIVE", "COD_EVE"]

        valid_consecutive = df["CONSECUTIVE"].notna() & (df["CONSECUTIVE"] != "SIN_DATO")

        df_with_id = df[valid_consecutive].drop_duplicates(subset=subset_columns)
        df_without_id = df[~valid_consecutive].drop_duplicates()

        df = pd.concat([df_with_id, df_without_id], ignore_index=True)
    else:
        df = df.drop_duplicates()

    after = len(df)
    removed = before - after

    return df, removed


def split_rejected_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separates valid rows from rejected rows.

    Comentario:
    Para el dashboard se requieren como mínimo año, semana, departamento y enfermedad.
    """

    invalid_mask = (
        df["ANO"].isna()
        | df["SEMANA"].isna()
        | df["enfermedad"].isna()
        | df["Departamento_ocurrencia"].isna()
        | (df["Departamento_ocurrencia"] == "SIN_DATO")
    )

    rejected_df = df[invalid_mask].copy()
    valid_df = df[~invalid_mask].copy()

    return valid_df, rejected_df


def build_aggregated_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Creates weekly aggregated dataset by disease and department."""

    agg = (
        df.groupby(
            ["enfermedad", "ANO", "SEMANA", "Departamento_ocurrencia"],
            observed=True,
            dropna=False,
        )
        .agg(
            total_casos=("enfermedad", "size"),
            confirmados=("confirmados", "sum"),
            hospitalizados=("hospitalizado", lambda value: (value == "Hospitalizado").sum()),
            fallecidos=("condicion_final", lambda value: (value == "Muerto").sum()),
            edad_promedio=("edad_anos", "mean"),
            pct_femenino=("SEXO", lambda value: (value == "F").mean() * 100),
            cod_dpto=("COD_DPTO_O", "first"),
        )
        .reset_index()
    )

    agg = agg.rename(
        columns={
            "ANO": "año",
            "Departamento_ocurrencia": "departamento",
        }
    )

    agg["month"] = agg.apply(
        lambda row: iso_week_to_month(row["año"], row["SEMANA"]),
        axis=1,
    )

    agg["edad_promedio"] = agg["edad_promedio"].round(2)
    agg["pct_femenino"] = agg["pct_femenino"].round(2)
    agg["risk_label"] = agg["total_casos"].apply(get_risk_label)

    agg = agg.sort_values(
        ["enfermedad", "departamento", "año", "SEMANA"],
        ascending=True,
    )

    return agg


def save_outputs(
    clean_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    quality_report_df: pd.DataFrame,
) -> None:
    """Saves ETL outputs in CSV and Parquet formats."""

    clean_csv_path = PROCESSED_DIR / "sivigila_clean.csv"
    clean_parquet_path = PROCESSED_DIR / "sivigila_clean.parquet"

    agg_csv_path = PROCESSED_DIR / "sivigila_agg.csv"
    agg_parquet_path = PROCESSED_DIR / "sivigila_agg.parquet"

    rejected_path = PROCESSED_DIR / "sivigila_rejected_rows.csv"
    quality_path = PROCESSED_DIR / "quality_report.csv"
    quality_json_path = PROCESSED_DIR / "quality_report.json"

    clean_df.to_csv(clean_csv_path, index=False, encoding="utf-8-sig")
    clean_df.to_parquet(clean_parquet_path, index=False)

    agg_df.to_csv(agg_csv_path, index=False, encoding="utf-8-sig")
    agg_df.to_parquet(agg_parquet_path, index=False)

    rejected_df.to_csv(rejected_path, index=False, encoding="utf-8-sig")
    quality_report_df.to_csv(quality_path, index=False, encoding="utf-8-sig")

    quality_json_path.write_text(
        json.dumps(
            quality_report_df.to_dict(orient="records"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for disease in clean_df["enfermedad"].dropna().unique():
        disease_slug = str(disease).lower().replace(" ", "_")
        disease_path = PROCESSED_DIR / f"sivigila_clean_{disease_slug}.csv"

        clean_df[clean_df["enfermedad"] == disease].to_csv(
            disease_path,
            index=False,
            encoding="utf-8-sig",
        )

    log.info("Clean CSV saved: %s", clean_csv_path)
    log.info("Clean Parquet saved: %s", clean_parquet_path)
    log.info("Aggregated CSV saved: %s", agg_csv_path)
    log.info("Aggregated Parquet saved: %s", agg_parquet_path)
    log.info("Rejected rows saved: %s", rejected_path)
    log.info("Quality report saved: %s", quality_path)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the complete preprocessing pipeline."""

    log.info("Starting strict preprocessing pipeline")

    frames = []

    for source in SOURCE_FILES:
        path = source["path"]

        if not path.exists():
            log.warning("Source file not found. Skipping: %s", path.name)
            continue

        raw_df = read_raw_file(path)

        standardized_df = standardize_columns(
            df=raw_df,
            disease=source["disease"],
            event_code=source["event_code"],
            year=source["year"],
            source=source["source"],
            filename=path.name,
        )

        cleaned_df = clean_dataframe(standardized_df)

        frames.append(cleaned_df)

        log.info(
            "Prepared %s - %s rows from %s",
            source["disease"],
            f"{len(cleaned_df):,}",
            path.name,
        )

    if not frames:
        raise FileNotFoundError(
            "No raw datasets were found. Run ingestion.py before preprocessing.py.",
        )

    combined_df = pd.concat(frames, ignore_index=True)

    combined_df, duplicates_removed = deduplicate_cases(combined_df)

    valid_df, rejected_df = split_rejected_rows(combined_df)

    agg_df = build_aggregated_dataset(valid_df)

    quality_report_df = build_quality_report(valid_df, rejected_df)

    extra_quality_rows = pd.DataFrame([
        {
            "metric": "duplicates_removed",
            "value": duplicates_removed,
        },
        {
            "metric": "total_aggregated_rows",
            "value": len(agg_df),
        },
    ])

    quality_report_df = pd.concat(
        [quality_report_df, extra_quality_rows],
        ignore_index=True,
    )

    save_outputs(
        clean_df=valid_df,
        agg_df=agg_df,
        rejected_df=rejected_df,
        quality_report_df=quality_report_df,
    )

    log.info("Preprocessing completed")
    log.info("Clean rows: %s", f"{len(valid_df):,}")
    log.info("Rejected rows: %s", f"{len(rejected_df):,}")
    log.info("Aggregated rows: %s", f"{len(agg_df):,}")

    return valid_df, agg_df


if __name__ == "__main__":
    clean, aggregated = run()

    print("\n── Casos por enfermedad ─────────────────────────────")
    print(clean.groupby("enfermedad").size().sort_values(ascending=False).to_string())

    print("\n── Dataset agregado ─────────────────────────────────")
    print(aggregated.head(10).to_string(index=False))

    print("\n── Archivos generados en data/processed/ ────────────")
    for file in sorted(PROCESSED_DIR.iterdir()):
        print(f"- {file.name}")