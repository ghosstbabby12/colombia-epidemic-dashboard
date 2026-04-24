"""
Extracción automática de datos desde fuentes abiertas colombianas.

Estrategia por fuente:
  - SIVIGILA (datos.gov.co) → Selenium (sitio requiere JS)
  - DANE                    → requests directo (Excel público)
  - IDEAM                   → requests directo (CSV público)

Uso:
  python src/data/ingestion.py
  python src/data/ingestion.py --fuente sivigila --año 2023
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── URLs y configuración de datasets ─────────────────────────────────────────

# datos.gov.co — IDs Socrata de cada enfermedad
SIVIGILA_DATASETS = {
    "dengue":      {"id": "ke8u-qixu", "cod_eve": 210},
    "chikungunya": {"id": "nu5z-zutz", "cod_eve": 217},
    "malaria":     {"id": "uayj-q8m7", "cod_eve": 460},
}

# DANE — Proyecciones poblacionales por municipio
DANE_URLS = [
    "https://www.dane.gov.co/files/investigaciones/poblacion/proyepobla06_20/DCD-area-proypoblacion-Mun-2018-2035.xlsx",
    "https://www.dane.gov.co/files/operaciones/PD/anexos/PD_Poblacion_municipio_area_sexo_edad_2018-2035.xlsx",
]

# IDEAM — Temperatura y precipitación histórica (portal DHIME)
IDEAM_CSV_URL = "http://dhime.ideam.gov.co/atencionciudadano/"


# ── FUENTE 1: SIVIGILA vía Selenium ──────────────────────────────────────────

def descargar_sivigila_selenium(enfermedad: str = "dengue", año: int = 2024) -> Path | None:
    """
    Automatiza la descarga del Excel SIVIGILA desde datos.gov.co usando Selenium.
    Abre Chrome en modo headless, navega al dataset y descarga el archivo.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        log.error("Instala: pip install selenium webdriver-manager")
        return None

    dataset_id = SIVIGILA_DATASETS[enfermedad]["id"]
    url = f"https://www.datos.gov.co/resource/{dataset_id}.csv?$limit=500000"

    # Configurar Chrome headless con carpeta de descarga
    download_dir = str(RAW_DIR.resolve())
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    })

    log.info("Iniciando Chrome headless para %s %d...", enfermedad, año)
    driver = None
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        # Intentar descarga directa del CSV via API Socrata con filtro de año
        csv_url = (
            f"https://www.datos.gov.co/resource/{dataset_id}.csv"
            f"?$where=ano={año}&$limit=500000"
        )
        driver.get(csv_url)
        time.sleep(5)

        # Verificar si descargó
        dest = RAW_DIR / f"sivigila_{enfermedad}_{año}_auto.csv"
        page_source = driver.page_source

        if "<!DOCTYPE html>" not in page_source and len(page_source) > 1000:
            dest.write_text(page_source, encoding="utf-8")
            log.info("Descargado: %s (%d bytes)", dest.name, dest.stat().st_size)
            return dest

        log.warning("La descarga vía Selenium no retornó CSV válido.")
        return None

    except Exception as e:
        log.error("Error Selenium: %s", e)
        return None
    finally:
        if driver:
            driver.quit()


def descargar_sivigila_requests(enfermedad: str = "dengue", año: int = 2024) -> Path | None:
    """
    Intenta descargar SIVIGILA directamente con requests (más rápido que Selenium).
    Funciona si datos.gov.co no bloquea el acceso programático.
    """
    dataset_id = SIVIGILA_DATASETS[enfermedad]["id"]
    dest = RAW_DIR / f"sivigila_{enfermedad}_{año}_auto.csv"

    if dest.exists() and dest.stat().st_size > 10_000:
        log.info("Usando caché: %s", dest.name)
        return dest

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)",
        "Accept": "text/csv,application/csv",
    }

    urls_a_intentar = [
        f"https://www.datos.gov.co/resource/{dataset_id}.csv?$where=ano={año}&$limit=500000",
        f"https://www.datos.gov.co/api/views/{dataset_id}/rows.csv?accessType=DOWNLOAD",
    ]

    for url in urls_a_intentar:
        try:
            log.info("Intentando: %s", url)
            r = requests.get(url, headers=headers, timeout=60, stream=True)
            if r.status_code == 200 and len(r.content) > 5000:
                dest.write_bytes(r.content)
                log.info("Descargado: %s (%s KB)", dest.name, dest.stat().st_size // 1024)
                return dest
            log.warning("HTTP %d en %s", r.status_code, url)
        except Exception as e:
            log.warning("requests falló: %s", e)

    return None


def cargar_sivigila_manual(enfermedad: str, año: int) -> pd.DataFrame | None:
    """
    Carga el archivo Excel descargado manualmente (flujo actual del equipo).
    Mapea el nombre del archivo al código de evento SIVIGILA.
    """
    cod_eve = SIVIGILA_DATASETS[enfermedad]["cod_eve"]

    candidatos = list(RAW_DIR.glob(f"*{cod_eve}*.xlsx")) + \
                 list(RAW_DIR.glob(f"*{enfermedad}*{año}*.csv")) + \
                 list(RAW_DIR.glob(f"*{enfermedad}*{año}*.xlsx"))

    if not candidatos:
        log.warning("No se encontró archivo manual para %s %d en data/raw/", enfermedad, año)
        return None

    path = candidatos[0]
    log.info("Cargando manual: %s", path.name)

    if path.suffix == ".csv":
        return pd.read_csv(path, dtype=str, low_memory=False)
    return pd.read_excel(path, dtype={"COD_DPTO_O": str, "COD_MUN_O": str})


# ── FUENTE 2: DANE — Población municipal ─────────────────────────────────────

def descargar_dane_poblacion() -> pd.DataFrame:
    """
    Descarga proyecciones poblacionales municipales del DANE.
    Necesario para calcular tasas de incidencia por 100,000 habitantes.
    """
    dest = RAW_DIR / "dane_poblacion.xlsx"

    if dest.exists():
        log.info("DANE desde caché: %s", dest.name)
    else:
        log.info("Descargando DANE proyecciones poblacionales...")
        descargado = False
        for url in DANE_URLS:
            try:
                r = requests.get(url, timeout=120, stream=True)
                if r.status_code == 200:
                    dest.write_bytes(r.content)
                    log.info("Guardado: %s (%d KB)", dest.name, dest.stat().st_size // 1024)
                    descargado = True
                    break
            except Exception as e:
                log.warning("DANE URL falló: %s", e)

        if not descargado:
            log.warning("No se pudo descargar DANE. Usando población estimada.")
            return _poblacion_referencia()

    try:
        df = pd.read_excel(dest, sheet_name=0, skiprows=9, dtype=str)
        log.info("DANE cargado: %d filas", len(df))
        return df
    except Exception as e:
        log.warning("Error leyendo DANE: %s. Usando referencia.", e)
        return _poblacion_referencia()


def _poblacion_referencia() -> pd.DataFrame:
    """Población departamental 2024 según DANE (dato fijo de referencia)."""
    return pd.DataFrame([
        {"departamento": "ANTIOQUIA",         "poblacion_2024": 6_967_737},
        {"departamento": "ATLANTICO",          "poblacion_2024": 2_722_128},
        {"departamento": "BOGOTA",             "poblacion_2024": 8_034_649},
        {"departamento": "BOLIVAR",            "poblacion_2024": 2_290_251},
        {"departamento": "BOYACA",             "poblacion_2024": 1_217_376},
        {"departamento": "CALDAS",             "poblacion_2024":   923_472},
        {"departamento": "CAQUETA",            "poblacion_2024":   522_287},
        {"departamento": "CAUCA",              "poblacion_2024": 1_479_992},
        {"departamento": "CESAR",              "poblacion_2024": 1_217_546},
        {"departamento": "CORDOBA",            "poblacion_2024": 1_861_792},
        {"departamento": "CUNDINAMARCA",       "poblacion_2024": 3_242_994},
        {"departamento": "CHOCO",              "poblacion_2024":   550_723},
        {"departamento": "HUILA",              "poblacion_2024": 1_168_918},
        {"departamento": "LA GUAJIRA",         "poblacion_2024": 1_107_589},
        {"departamento": "MAGDALENA",          "poblacion_2024": 1_441_475},
        {"departamento": "META",               "poblacion_2024": 1_115_059},
        {"departamento": "NARIÑO",             "poblacion_2024": 1_680_008},
        {"departamento": "NORTE SANTANDER",    "poblacion_2024": 1_612_467},
        {"departamento": "QUINDIO",            "poblacion_2024":   577_802},
        {"departamento": "RISARALDA",          "poblacion_2024":   994_532},
        {"departamento": "SANTANDER",          "poblacion_2024": 2_264_376},
        {"departamento": "SUCRE",              "poblacion_2024":   988_847},
        {"departamento": "TOLIMA",             "poblacion_2024": 1_315_285},
        {"departamento": "VALLE",              "poblacion_2024": 4_748_024},
        {"departamento": "ARAUCA",             "poblacion_2024":   319_980},
        {"departamento": "CASANARE",           "poblacion_2024":   460_147},
        {"departamento": "PUTUMAYO",           "poblacion_2024":   371_457},
        {"departamento": "AMAZONAS",           "poblacion_2024":    84_153},
        {"departamento": "GUAINIA",            "poblacion_2024":    51_564},
        {"departamento": "GUAVIARE",           "poblacion_2024":   123_480},
        {"departamento": "VAUPES",             "poblacion_2024":    48_320},
        {"departamento": "VICHADA",            "poblacion_2024":   118_451},
        {"departamento": "SAN ANDRES",         "poblacion_2024":    76_442},
    ])


# ── FUENTE 3: IDEAM — Datos climáticos ───────────────────────────────────────

def generar_clima_referencia(departamentos: list[str] | None = None) -> pd.DataFrame:
    """
    Genera variables climáticas de referencia por departamento.
    Basado en promedios históricos IDEAM (temperatura media anual y
    precipitación media por región climática de Colombia).

    Reemplazar con descarga real del DHIME-IDEAM cuando se tenga acceso.
    """
    import numpy as np

    clima_base = {
        "VALLE":            {"temp": 24.0, "precip": 1650, "humedad": 78},
        "SANTANDER":        {"temp": 23.5, "precip": 1900, "humedad": 74},
        "TOLIMA":           {"temp": 26.5, "precip": 1450, "humedad": 70},
        "HUILA":            {"temp": 25.0, "precip": 1350, "humedad": 68},
        "ANTIOQUIA":        {"temp": 22.0, "precip": 2400, "humedad": 80},
        "CUNDINAMARCA":     {"temp": 19.0, "precip": 1000, "humedad": 72},
        "BOLIVAR":          {"temp": 28.5, "precip": 1100, "humedad": 82},
        "CAUCA":            {"temp": 20.0, "precip": 1800, "humedad": 79},
        "RISARALDA":        {"temp": 21.0, "precip": 2600, "humedad": 83},
        "NORTE SANTANDER":  {"temp": 24.0, "precip": 1200, "humedad": 71},
        "ATLANTICO":        {"temp": 28.0, "precip":  850, "humedad": 77},
        "CESAR":            {"temp": 29.0, "precip": 1050, "humedad": 75},
        "MAGDALENA":        {"temp": 28.5, "precip":  950, "humedad": 76},
        "CORDOBA":          {"temp": 28.0, "precip": 1400, "humedad": 80},
        "SUCRE":            {"temp": 28.5, "precip": 1100, "humedad": 79},
        "META":             {"temp": 26.0, "precip": 3200, "humedad": 82},
        "NARIÑO":           {"temp": 18.0, "precip": 2200, "humedad": 81},
        "LA GUAJIRA":       {"temp": 29.5, "precip":  450, "humedad": 65},
        "CHOCO":            {"temp": 27.0, "precip": 8000, "humedad": 90},
        "BOYACA":           {"temp": 14.0, "precip": 1100, "humedad": 69},
    }

    departamentos = departamentos or list(clima_base.keys())
    rng = np.random.default_rng(42)
    rows = []

    for dept in departamentos:
        base = clima_base.get(dept, {"temp": 25.0, "precip": 1500, "humedad": 75})
        for semana in range(1, 53):
            # Estacionalidad: más lluvia en semanas 15-30 y 40-50
            factor_lluvia = 1.4 if semana in range(15, 31) or semana in range(40, 51) else 0.7
            rows.append({
                "departamento": dept,
                "semana": semana,
                "año": 2024,
                "temperatura_media": round(base["temp"] + rng.uniform(-2, 2), 1),
                "precipitacion_mm": round(base["precip"] / 52 * factor_lluvia + rng.uniform(0, 20), 1),
                "humedad_relativa": round(min(99, base["humedad"] + rng.uniform(-5, 5)), 1),
            })

    return pd.DataFrame(rows)


# ── Orquestador principal ─────────────────────────────────────────────────────

def run(año: int = 2024, forzar_descarga: bool = False) -> None:
    log.info("=== Inicio de ingesta (año %d) ===", año)

    # ── SIVIGILA ──
    for enfermedad in SIVIGILA_DATASETS:
        dest_csv = RAW_DIR / f"sivigila_{enfermedad}_{año}_auto.csv"

        if dest_csv.exists() and not forzar_descarga:
            log.info("SIVIGILA %s ya existe, omitiendo descarga.", enfermedad)
            continue

        # Intento 1: requests directo
        resultado = descargar_sivigila_requests(enfermedad, año)

        # Intento 2: Selenium (si requests falló)
        if resultado is None:
            log.info("Intentando con Selenium para %s...", enfermedad)
            resultado = descargar_sivigila_selenium(enfermedad, año)

        # Intento 3: archivo manual del equipo (fallback)
        if resultado is None:
            df_manual = cargar_sivigila_manual(enfermedad, año)
            if df_manual is not None:
                out = RAW_DIR / f"sivigila_{enfermedad}_{año}_manual.csv"
                df_manual.to_csv(out, index=False)
                log.info("Usando archivo manual → %s", out.name)

    # ── DANE ──
    df_dane = descargar_dane_poblacion()
    dane_path = RAW_DIR / "dane_poblacion_referencia.csv"
    if "departamento" in df_dane.columns:
        df_dane.to_csv(dane_path, index=False)
        log.info("Guardado: %s", dane_path.name)

    # ── IDEAM ──
    df_clima = generar_clima_referencia()
    clima_path = RAW_DIR / f"ideam_clima_{año}.csv"
    df_clima.to_csv(clima_path, index=False)
    log.info("Guardado: %s (%d filas)", clima_path.name, len(df_clima))

    log.info("=== Ingesta completada ===")
    log.info("Archivos en data/raw/:")
    for f in sorted(RAW_DIR.iterdir()):
        log.info("  %s (%s KB)", f.name, f.stat().st_size // 1024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--año", type=int, default=2024)
    parser.add_argument("--forzar", action="store_true", help="Re-descargar aunque exista caché")
    args = parser.parse_args()
    run(año=args.año, forzar_descarga=args.forzar)
