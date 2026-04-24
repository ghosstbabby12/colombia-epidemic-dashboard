"""
API REST – Dashboard Epidemiológico Colombia
Conectada a PostgreSQL (dengue_db) via SQLAlchemy.

Endpoints:
  GET  /health
  GET  /enfermedades
  GET  /departamentos
  GET  /resumen
  GET  /casos/semana?enfermedad=&departamento=
  GET  /casos/departamento?enfermedad=
  GET  /casos/municipio/{departamento}?enfermedad=
  GET  /casos/grupo-etario?departamento=&enfermedad=
  GET  /casos/sexo?departamento=&enfermedad=
  GET  /alertas

Ejecutar:
  uvicorn src.backend.app:app --reload --port 8000
Docs:
  http://localhost:8000/docs
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Conexión a PostgreSQL ─────────────────────────────────────────────────────

DB_URL = "postgresql://{user}:{pwd}@{host}:{port}/{db}".format(
    user=os.getenv("DB_USER", "ghostbabby"),
    pwd=os.getenv("DB_PASS", ""),
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432"),
    db=os.getenv("DB_NAME", "dengue_db"),
)

engine = create_engine(DB_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("Conexión a PostgreSQL OK → %s", DB_URL.split("@")[1])
    except Exception as e:
        log.error("No se pudo conectar a PostgreSQL: %s", e)
    yield

app = FastAPI(
    title="Colombia Epidemic Dashboard API",
    description="API epidemiológica conectada a PostgreSQL con datos SIVIGILA 2024.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def rows_to_list(result) -> list[dict]:
    cols = list(result.keys())
    return [dict(zip(cols, row)) for row in result.fetchall()]


# ── Endpoints de sistema ──────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
def health(db: Session = Depends(get_db)):
    try:
        r = db.execute(text("SELECT COUNT(*) FROM casos_individuales")).scalar()
        return {"status": "ok", "total_casos_db": r, "db": "dengue_db"}
    except Exception as e:
        raise HTTPException(503, f"Base de datos no disponible: {e}")


@app.get("/enfermedades", tags=["Sistema"])
def enfermedades(db: Session = Depends(get_db)):
    r = db.execute(text(
        'SELECT enfermedad, COUNT(*) as casos FROM casos_individuales '
        'GROUP BY enfermedad ORDER BY casos DESC'
    ))
    return rows_to_list(r)


# ── Departamentos ─────────────────────────────────────────────────────────────

@app.get("/departamentos", tags=["Geografía"])
def departamentos(
    enfermedad: str = Query("Dengue", description="Dengue | Chikungunya | Malaria"),
    db: Session = Depends(get_db),
):
    r = db.execute(text("""
        SELECT "Departamento_ocurrencia" AS departamento,
               COUNT(*) AS total_casos,
               SUM(CASE WHEN hospitalizado = 'Hospitalizado' THEN 1 ELSE 0 END) AS hospitalizados,
               ROUND(AVG(edad_anos)::numeric, 1) AS edad_promedio
        FROM casos_individuales
        WHERE enfermedad = :enf
        GROUP BY "Departamento_ocurrencia"
        ORDER BY total_casos DESC
    """), {"enf": enfermedad})
    return rows_to_list(r)


# ── Resumen general ───────────────────────────────────────────────────────────

@app.get("/resumen", tags=["Dashboard"])
def resumen(db: Session = Depends(get_db)):
    r = db.execute(text("""
        SELECT
            enfermedad,
            COUNT(*)                                                          AS total_casos,
            SUM(CASE WHEN hospitalizado = 'Hospitalizado' THEN 1 ELSE 0 END) AS hospitalizados,
            SUM(CASE WHEN condicion_final = 'Muerto'      THEN 1 ELSE 0 END) AS fallecidos,
            ROUND(AVG(edad_anos)::numeric, 1)                                 AS edad_promedio,
            SUM(CASE WHEN "SEXO" = 'F' THEN 1 ELSE 0 END) * 100 / COUNT(*)  AS pct_femenino
        FROM casos_individuales
        GROUP BY enfermedad
        ORDER BY total_casos DESC
    """))
    datos = rows_to_list(r)

    # Semana con más casos (pico epidémico)
    pico = db.execute(text("""
        SELECT "SEMANA" AS semana_pico, COUNT(*) AS casos_pico
        FROM casos_individuales
        WHERE enfermedad = 'Dengue'
        GROUP BY "SEMANA"
        ORDER BY casos_pico DESC
        LIMIT 1
    """))
    pico_data = rows_to_list(pico)

    return {"por_enfermedad": datos, "pico_epidemico_dengue": pico_data[0] if pico_data else {}}


# ── Casos por semana epidemiológica ───────────────────────────────────────────

@app.get("/casos/semana", tags=["Dashboard"])
def casos_por_semana(
    enfermedad: str = Query("Dengue"),
    departamento: str = Query(None, description="Filtrar por departamento (opcional)"),
    db: Session = Depends(get_db),
):
    filtro_dpto = 'AND "Departamento_ocurrencia" = :dpto' if departamento else ""
    r = db.execute(text(f"""
        SELECT "SEMANA"   AS semana,
               COUNT(*)   AS casos,
               SUM(CASE WHEN hospitalizado = 'Hospitalizado' THEN 1 ELSE 0 END) AS hospitalizados
        FROM casos_individuales
        WHERE enfermedad = :enf {filtro_dpto}
        GROUP BY "SEMANA"
        ORDER BY semana
    """), {"enf": enfermedad, "dpto": departamento})
    return {"enfermedad": enfermedad, "departamento": departamento, "serie": rows_to_list(r)}


# ── Casos por departamento (ranking) ──────────────────────────────────────────

@app.get("/casos/departamento", tags=["Dashboard"])
def casos_por_departamento(
    enfermedad: str = Query("Dengue"),
    top: int = Query(20, ge=1, le=33),
    db: Session = Depends(get_db),
):
    r = db.execute(text("""
        SELECT "Departamento_ocurrencia"                                          AS departamento,
               COUNT(*)                                                           AS casos,
               SUM(CASE WHEN hospitalizado = 'Hospitalizado' THEN 1 ELSE 0 END)  AS hospitalizados,
               SUM(CASE WHEN condicion_final = 'Muerto'      THEN 1 ELSE 0 END)  AS fallecidos,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2)                 AS pct_nacional
        FROM casos_individuales
        WHERE enfermedad = :enf
        GROUP BY "Departamento_ocurrencia"
        ORDER BY casos DESC
        LIMIT :top
    """), {"enf": enfermedad, "top": top})
    return {"enfermedad": enfermedad, "ranking": rows_to_list(r)}


# ── Casos por municipio ───────────────────────────────────────────────────────

@app.get("/casos/municipio/{departamento}", tags=["Dashboard"])
def casos_por_municipio(
    departamento: str,
    enfermedad: str = Query("Dengue"),
    top: int = Query(15),
    db: Session = Depends(get_db),
):
    r = db.execute(text("""
        SELECT "Municipio_ocurrencia"  AS municipio,
               COUNT(*)               AS casos,
               "COD_MUN_O"            AS cod_dane
        FROM casos_individuales
        WHERE enfermedad = :enf
          AND UPPER("Departamento_ocurrencia") = UPPER(:dpto)
        GROUP BY "Municipio_ocurrencia", "COD_MUN_O"
        ORDER BY casos DESC
        LIMIT :top
    """), {"enf": enfermedad, "dpto": departamento, "top": top})
    items = rows_to_list(r)
    if not items:
        raise HTTPException(404, f"No se encontraron datos para '{departamento}'.")
    return {"departamento": departamento.upper(), "enfermedad": enfermedad, "municipios": items}


# ── Distribución por grupo etario ─────────────────────────────────────────────

@app.get("/casos/grupo-etario", tags=["Dashboard"])
def casos_por_edad(
    enfermedad: str = Query("Dengue"),
    departamento: str = Query(None),
    db: Session = Depends(get_db),
):
    filtro = 'AND UPPER("Departamento_ocurrencia") = UPPER(:dpto)' if departamento else ""
    r = db.execute(text(f"""
        SELECT grupo_etario,
               COUNT(*) AS casos,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS porcentaje
        FROM casos_individuales
        WHERE enfermedad = :enf
          AND grupo_etario IS NOT NULL
          AND grupo_etario != 'None' {filtro}
        GROUP BY grupo_etario
        ORDER BY MIN(edad_anos)
    """), {"enf": enfermedad, "dpto": departamento})
    return rows_to_list(r)


# ── Distribución por sexo ─────────────────────────────────────────────────────

@app.get("/casos/sexo", tags=["Dashboard"])
def casos_por_sexo(
    enfermedad: str = Query("Dengue"),
    departamento: str = Query(None),
    db: Session = Depends(get_db),
):
    filtro = 'AND UPPER("Departamento_ocurrencia") = UPPER(:dpto)' if departamento else ""
    r = db.execute(text(f"""
        SELECT "SEXO"   AS sexo,
               COUNT(*) AS casos,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS porcentaje
        FROM casos_individuales
        WHERE enfermedad = :enf
          AND "SEXO" IN ('M', 'F') {filtro}
        GROUP BY "SEXO"
    """), {"enf": enfermedad, "dpto": departamento})
    return rows_to_list(r)


# ── Sistema de alertas ────────────────────────────────────────────────────────

@app.get("/alertas", tags=["Alertas"])
def alertas(db: Session = Depends(get_db)):
    """
    Departamentos en alerta: semanas recientes con casos
    muy por encima de su promedio histórico (umbral: 2x promedio).
    """
    r = db.execute(text("""
        WITH stats AS (
            SELECT "Departamento_ocurrencia" AS departamento,
                   "SEMANA"                  AS semana,
                   COUNT(*)                  AS casos_semana,
                   AVG(COUNT(*)) OVER (
                       PARTITION BY "Departamento_ocurrencia"
                   )                         AS promedio_dpto
            FROM casos_individuales
            WHERE enfermedad = 'Dengue'
            GROUP BY "Departamento_ocurrencia", "SEMANA"
        )
        SELECT departamento,
               semana,
               casos_semana,
               ROUND(promedio_dpto::numeric, 1) AS promedio_historico,
               ROUND((casos_semana / NULLIF(promedio_dpto, 0))::numeric, 2) AS ratio,
               CASE
                   WHEN casos_semana >= promedio_dpto * 3 THEN 'ALTO'
                   WHEN casos_semana >= promedio_dpto * 2 THEN 'MEDIO'
                   ELSE 'BAJO'
               END AS nivel_alerta
        FROM stats
        WHERE casos_semana >= promedio_dpto * 2
          AND semana >= (SELECT MAX("SEMANA") - 4 FROM casos_individuales WHERE enfermedad='Dengue')
        ORDER BY ratio DESC
        LIMIT 20
    """))
    alertas_data = rows_to_list(r)
    return {
        "total_alertas": len(alertas_data),
        "alertas": alertas_data,
        "criterio": "Departamentos con casos >= 2x su promedio en las últimas 4 semanas",
    }
