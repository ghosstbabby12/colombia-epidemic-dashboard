"""
Pipeline ETL para datos SIVIGILA reales.

Archivos de entrada (data/raw/):
  - Datos_2024_210.xlsx  → Dengue             (309,627 filas)
  - Datos_2024_217.xlsx  → Chikungunya         (54 filas)
  - Datos_2024_460.xlsx  → Malaria asociada    (2,249 filas)

Salidas (data/processed/):
  - sivigila_clean.parquet    → Casos individuales limpios
  - sivigila_agg.parquet      → Agregado por dpto + semana + enfermedad
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

# ── Archivos fuente ───────────────────────────────────────────────────────────
ARCHIVOS = {
    "Dengue":      RAW_DIR / "Datos_2024_210.xlsx",
    "Chikungunya": RAW_DIR / "Datos_2024_217.xlsx",
    "Malaria":     RAW_DIR / "Datos_2024_460.xlsx",
}

# ── Columnas que vamos a conservar ────────────────────────────────────────────
COLS_UTILES = [
    "CONSECUTIVE",            # ID único del caso
    "COD_EVE",                # Código SIVIGILA del evento
    "Nombre_evento",          # Nombre de la enfermedad
    "FEC_NOT",                # Fecha de notificación
    "SEMANA",                 # Semana epidemiológica
    "ANO",                    # Año
    "INI_SIN",                # Inicio de síntomas
    "EDAD",                   # Edad del paciente
    "UNI_MED",                # Unidad de medida (1=años, 2=meses, 3=días)
    "SEXO",                   # Sexo (M/F)
    "estrato",                # Estrato socioeconómico
    "TIP_SS",                 # Tipo de seguridad social (C=Contributivo, S=Subsidiado...)
    "TIP_CAS",                # Tipo de caso (1=Sosp, 2=Prob, 3=Lab, 4=Clín)
    "PAC_HOS",                # Hospitalizado (1=Sí, 2=No)
    "CON_FIN",                # Condición final (1=Vivo, 2=Muerto)
    "GP_GESTAN",              # Gestante (1=Sí, 2=No)
    "AREA",                   # Área (1=Cabecera, 2=Rural disperso, 3=Centro poblado)
    "COD_DPTO_O",             # Código DANE departamento de ocurrencia
    "COD_MUN_O",              # Código DANE municipio de ocurrencia
    "Departamento_ocurrencia",
    "Municipio_ocurrencia",
    "Departamento_residencia",
    "Municipio_residencia",
    "nom_est_f_caso",         # Estado final: Probable, Confirmado por laboratorio...
    "confirmados",            # Bandera: 1=confirmado
]

# ── Mapas de decodificación ───────────────────────────────────────────────────
TIP_CAS_MAP = {1: "Sospechoso", 2: "Probable", 3: "Confirmado_Lab", 4: "Confirmado_Clínico"}
CON_FIN_MAP = {1: "Vivo", 2: "Muerto"}
PAC_HOS_MAP = {1: "Hospitalizado", 2: "No hospitalizado"}
AREA_MAP    = {1: "Cabecera municipal", 2: "Rural disperso", 3: "Centro poblado"}
UNI_MED_MAP = {1: "Años", 2: "Meses", 3: "Días"}
TIP_SS_MAP  = {
    "C": "Contributivo", "S": "Subsidiado", "E": "Excepción",
    "P": "Especial", "I": "Indeterminado", "N": "No asegurado",
}


# ── Funciones de limpieza ─────────────────────────────────────────────────────

def cargar_archivo(path: Path, enfermedad: str) -> pd.DataFrame:
    log.info("Cargando %s (%s)...", path.name, enfermedad)
    df = pd.read_excel(path, dtype={"COD_DPTO_O": str, "COD_MUN_O": str})

    # Conservar solo columnas útiles que existan
    cols = [c for c in COLS_UTILES if c in df.columns]
    df = df[cols].copy()
    df["enfermedad"] = enfermedad
    log.info("  → %d filas, %d columnas", *df.shape)
    return df


def limpiar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    # Fechas
    for col in ["FEC_NOT", "INI_SIN"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Enteros
    for col in ["SEMANA", "ANO", "EDAD", "COD_EVE", "TIP_CAS", "PAC_HOS",
                "CON_FIN", "GP_GESTAN", "AREA", "UNI_MED", "confirmados"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Estrato: puede ser numérico o texto
    if "estrato" in df.columns:
        df["estrato"] = pd.to_numeric(df["estrato"], errors="coerce").astype("Int64")

    # Texto
    for col in ["SEXO", "TIP_SS", "Departamento_ocurrencia", "Municipio_ocurrencia",
                "Departamento_residencia", "Municipio_residencia", "nom_est_f_caso"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    return df


def agregar_columnas_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    # Edad en años normalizada (convierte meses y días a años)
    if "EDAD" in df.columns and "UNI_MED" in df.columns:
        edad_anos = df["EDAD"].astype(float).copy()
        edad_anos[df["UNI_MED"] == 2] = df.loc[df["UNI_MED"] == 2, "EDAD"] / 12
        edad_anos[df["UNI_MED"] == 3] = df.loc[df["UNI_MED"] == 3, "EDAD"] / 365
        df["edad_anos"] = edad_anos.round(1)

    # Grupo etario
    if "edad_anos" in df.columns:
        df["grupo_etario"] = pd.cut(
            df["edad_anos"],
            bins=[-1, 4, 14, 29, 44, 59, np.inf],
            labels=["0-4", "5-14", "15-29", "30-44", "45-59", "60+"],
        )

    # Decodificar variables categóricas
    if "TIP_CAS" in df.columns:
        df["tipo_caso"] = df["TIP_CAS"].map(TIP_CAS_MAP)
    if "CON_FIN" in df.columns:
        df["condicion_final"] = df["CON_FIN"].map(CON_FIN_MAP)
    if "PAC_HOS" in df.columns:
        df["hospitalizado"] = df["PAC_HOS"].map(PAC_HOS_MAP)
    if "AREA" in df.columns:
        df["area_residencia"] = df["AREA"].map(AREA_MAP)
    if "TIP_SS" in df.columns:
        df["regimen_salud"] = df["TIP_SS"].map(TIP_SS_MAP).fillna(df["TIP_SS"])

    # Semana de inicio de síntomas
    if "INI_SIN" in df.columns:
        df["semana_inicio_sintomas"] = df["INI_SIN"].dt.isocalendar().week.astype("Int64")

    # Tiempo de notificación (días entre inicio síntomas y notificación)
    if "FEC_NOT" in df.columns and "INI_SIN" in df.columns:
        df["dias_notificacion"] = (df["FEC_NOT"] - df["INI_SIN"]).dt.days

    return df


def agregar_por_dpto_semana(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea el dataset agregado que usará el modelo ML y el dashboard.
    Nivel: enfermedad + departamento + semana epidemiológica.
    """
    agg = (
        df.groupby(["enfermedad", "ANO", "SEMANA", "Departamento_ocurrencia"], observed=True)
        .agg(
            total_casos=("CONSECUTIVE", "count"),
            confirmados=("confirmados", "sum"),
            hospitalizados=("PAC_HOS", lambda x: (x == 1).sum()),
            fallecidos=("CON_FIN", lambda x: (x == 2).sum()),
            edad_promedio=("edad_anos", "mean"),
            pct_femenino=("SEXO", lambda x: (x == "F").mean() * 100),
            cod_dpto=("COD_DPTO_O", "first"),
        )
        .reset_index()
    )
    agg = agg.rename(columns={"Departamento_ocurrencia": "departamento", "ANO": "año"})
    agg["edad_promedio"] = agg["edad_promedio"].round(1)
    agg["pct_femenino"] = agg["pct_femenino"].round(1)
    agg = agg.sort_values(["enfermedad", "departamento", "año", "SEMANA"])
    return agg


# ── Pipeline principal ────────────────────────────────────────────────────────

def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("=== Inicio del pipeline ETL SIVIGILA ===")

    frames = []
    for enfermedad, path in ARCHIVOS.items():
        if not path.exists():
            log.warning("Archivo no encontrado, omitiendo: %s", path)
            continue
        df = cargar_archivo(path, enfermedad)
        df = limpiar_tipos(df)
        df = agregar_columnas_derivadas(df)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No se encontraron archivos en data/raw/. Verifica los nombres.")

    df_clean = pd.concat(frames, ignore_index=True)
    log.info("Dataset combinado: %d filas, %d columnas", *df_clean.shape)

    # Guardar casos individuales
    out_clean = PROCESSED_DIR / "sivigila_clean.parquet"
    df_clean.to_parquet(out_clean, index=False)
    log.info("Guardado: %s", out_clean)

    # Guardar dataset agregado
    df_agg = agregar_por_dpto_semana(df_clean)
    out_agg = PROCESSED_DIR / "sivigila_agg.parquet"
    df_agg.to_parquet(out_agg, index=False)
    log.info("Guardado: %s  (%d filas)", out_agg, len(df_agg))

    log.info("=== ETL completado ===")
    return df_clean, df_agg


if __name__ == "__main__":
    df_clean, df_agg = run()

    print("\n── Resumen dataset limpio ──────────────────────────────")
    print(df_clean.dtypes.to_string())
    print(f"\nFilas: {len(df_clean):,}")

    print("\n── Muestra dataset agregado ────────────────────────────")
    print(df_agg.head(10).to_string())

    print("\n── Casos por enfermedad ────────────────────────────────")
    print(df_clean.groupby("enfermedad")["CONSECUTIVE"].count().to_string())

    print("\n── Top 10 departamentos por casos de Dengue ────────────")
    dengue = df_clean[df_clean["enfermedad"] == "Dengue"]
    print(dengue.groupby("Departamento_ocurrencia")["CONSECUTIVE"]
          .count().sort_values(ascending=False).head(10).to_string())
