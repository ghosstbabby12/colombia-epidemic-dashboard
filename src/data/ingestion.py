"""
Ingesta de datos desde fuentes abiertas colombianas.

Fuentes:
  - datos.gov.co (SIVIGILA / INS) via API Socrata
  - DANE (proyecciones poblacionales)
  - IDEAM (variables climáticas)
"""

import logging
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from sodapy import Socrata

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuración de recursos Socrata (datos.gov.co) ─────────────────────────
SOCRATA_DOMAIN = "www.datos.gov.co"
SOCRATA_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN", None)  # Opcional pero recomendado

# IDs de dataset en datos.gov.co
# Actualizar con el ID correcto si el dataset cambia de versión.
DATASETS = {
    "dengue": "gftq-kdpd",           # Notificaciones SIVIGILA – Dengue
    "dengue_grave": "cqv4-eis4",     # Dengue grave
}

DANE_POBLACION_URL = (
    "https://www.dane.gov.co/files/investigaciones/poblacion/"
    "proyepobla06_20/Municipal_area_1985-2020.xlsx"
)


# ── Funciones de descarga ─────────────────────────────────────────────────────

def fetch_sivigila_dengue(year: int = 2023, limit: int = 100_000) -> pd.DataFrame:
    """
    Descarga notificaciones de dengue desde datos.gov.co (API Socrata).

    Parámetros
    ----------
    year   : Año epidemiológico a descargar.
    limit  : Máximo de filas a recuperar por llamada.
    """
    log.info("Descargando datos SIVIGILA – Dengue (%d)...", year)

    client = Socrata(SOCRATA_DOMAIN, SOCRATA_APP_TOKEN, timeout=60)

    try:
        results = client.get(
            DATASETS["dengue"],
            where=f"año={year}",
            limit=limit,
        )
        df = pd.DataFrame.from_records(results)
        log.info("  → %d registros descargados.", len(df))
    except Exception as exc:
        log.warning("Fallo API Socrata (%s). Usando datos de muestra.", exc)
        df = _sample_dengue_data(year)
    finally:
        client.close()

    return df


def fetch_dane_poblacion() -> pd.DataFrame:
    """
    Descarga proyecciones poblacionales municipales del DANE.
    Usa caché local si el archivo ya existe.
    """
    dest = RAW_DIR / "dane_poblacion_municipal.xlsx"

    if dest.exists():
        log.info("Cargando DANE desde caché: %s", dest)
    else:
        log.info("Descargando proyecciones poblacionales DANE...")
        try:
            resp = requests.get(DANE_POBLACION_URL, timeout=120)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            log.info("  → Guardado en %s", dest)
        except Exception as exc:
            log.warning("No se pudo descargar DANE (%s). Continuando sin él.", exc)
            return pd.DataFrame()

    return pd.read_excel(dest, sheet_name=0, header=9)


def fetch_ideam_clima(departamentos: list[str] | None = None) -> pd.DataFrame:
    """
    Retorna datos climáticos sintéticos con la estructura esperada.
    Reemplazar con llamada real a la API de IDEAM cuando esté disponible.

    Columnas: departamento, semana_epidemiologica, año, temperatura_media,
              precipitacion_mm, humedad_relativa
    """
    log.info("Generando datos climáticos de referencia (placeholder IDEAM)...")

    departamentos = departamentos or [
        "ANTIOQUIA", "VALLE DEL CAUCA", "CUNDINAMARCA",
        "BOLIVAR", "SANTANDER", "ATLANTICO", "TOLIMA",
        "HUILA", "META", "CESAR",
    ]

    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    for dept in departamentos:
        for semana in range(1, 53):
            rows.append({
                "departamento": dept,
                "semana_epidemiologica": semana,
                "año": 2023,
                "temperatura_media": round(rng.uniform(20, 35), 1),
                "precipitacion_mm": round(rng.uniform(0, 300), 1),
                "humedad_relativa": round(rng.uniform(50, 95), 1),
            })

    return pd.DataFrame(rows)


# ── Datos de muestra (fallback) ───────────────────────────────────────────────

def _sample_dengue_data(year: int) -> pd.DataFrame:
    """Genera un dataset mínimo con la estructura de SIVIGILA para pruebas."""
    import numpy as np

    rng = np.random.default_rng(0)
    departamentos = [
        "ANTIOQUIA", "VALLE DEL CAUCA", "CUNDINAMARCA",
        "BOLIVAR", "SANTANDER", "ATLANTICO",
    ]
    n = 500
    return pd.DataFrame({
        "año": [str(year)] * n,
        "semana": rng.integers(1, 53, n).astype(str),
        "departamento": rng.choice(departamentos, n),
        "municipio": ["MUNICIPIO_" + str(i % 30) for i in range(n)],
        "casos": rng.integers(1, 200, n).astype(str),
        "tipo": rng.choice(["DENGUE", "DENGUE CON SIGNOS DE ALARMA"], n),
        "edad": rng.integers(1, 80, n).astype(str),
        "sexo": rng.choice(["M", "F"], n),
    })


# ── Orquestador principal ─────────────────────────────────────────────────────

def run(year: int = 2023) -> None:
    log.info("=== Inicio de ingesta de datos (año %d) ===", year)

    df_dengue = fetch_sivigila_dengue(year)
    df_dengue.to_csv(RAW_DIR / f"sivigila_dengue_{year}.csv", index=False)
    log.info("Guardado: sivigila_dengue_%d.csv", year)

    df_poblacion = fetch_dane_poblacion()
    if not df_poblacion.empty:
        df_poblacion.to_csv(RAW_DIR / "dane_poblacion.csv", index=False)
        log.info("Guardado: dane_poblacion.csv")

    df_clima = fetch_ideam_clima()
    df_clima.to_csv(RAW_DIR / "ideam_clima_2023.csv", index=False)
    log.info("Guardado: ideam_clima_2023.csv")

    log.info("=== Ingesta completada ===")


if __name__ == "__main__":
    run(year=2023)
